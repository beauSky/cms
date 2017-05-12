/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: hsc/kisslovecsh@foxmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <conn/cms_http_s.h>
#include <log/cms_log.h>
#include <enc/cms_sha1.h>
#include <ev/cms_ev.h>
#include <conn/cms_conn_var.h>
#include <common/cms_utility.h>
#include <taskmgr/cms_task_mgr.h>
#include <static/cms_static.h>


CHttpServer::CHttpServer(CReaderWriter *rw,bool isTls)
{
	char remote[23] = {0};
	rw->remoteAddr(remote,sizeof(remote));
	mremoteAddr = remote;
	size_t pos = mremoteAddr.find(":");
	if (pos == string::npos)
	{
		mremoteIP = mremoteAddr;
	}
	else
	{
		mremoteIP = mremoteAddr.substr(0,pos);
	}
	mrdBuff = new CBufferReader(rw,128*1024);
	assert(mrdBuff);
	mwrBuff = new CBufferWriter(rw,128*1024);
	assert(mwrBuff);
	mrw = rw;
	mhttp = new CHttp(this,mrdBuff,mwrBuff,rw,mremoteAddr,false,isTls);
	mloop = NULL;
	mwatcherReadIO = NULL;
	mwatcherWriteIO = NULL;
	mllIdx = 0;
	misDecodeHeader = false;
	misWebSocket = false;
	mflvTrans = new CFlvTransmission(mhttp);
	mbinaryWriter = NULL;
	misAddConn = false;
	misFlvRequest = false;
	misStop = false;

	mspeedTick = 0;
}

CHttpServer::~CHttpServer()
{
	logs->debug("######### %s [CHttpServer::~CHttpServer] http enter ",
		mremoteAddr.c_str());
	if (mloop)
	{
		if (mwatcherReadIO)
		{
			ev_io_stop(mloop,mwatcherReadIO);
			delete mwatcherReadIO;
			logs->debug("######### %s [CHttpServer::~CHttpServer] stop read io ",
				mremoteAddr.c_str());
		}
		if (mwatcherWriteIO)
		{
			ev_io_stop(mloop,mwatcherWriteIO);
			delete mwatcherWriteIO;

			logs->debug("######### %s [CHttpServer::~CHttpServer] stop write io ",
				mremoteAddr.c_str());
		}
	}
	delete mflvTrans;
	delete mhttp;
	delete mrdBuff;
	delete mwrBuff;
	if (mbinaryWriter)
	{
		delete mbinaryWriter;
	}
	mrw->close();
	delete mrw;
}

int CHttpServer::doit()
{
	if (!mhttp->run())
	{
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CHttpServer::handleEv(FdEvents *fe)
{
	if (misStop)
	{
		return CMS_ERROR;
	}

	if (fe->events & EventWrite || fe->events & EventWait2Write)
	{
		if (fe->events & EventWait2Write && fe->watcherWCmsTimer !=  mhttp->cmsTimer2Write())
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		else if (fe->events & EventWrite && mwatcherWriteIO != fe->watcherWriteIO)
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		return doWrite(fe->events & EventWait2Write);
	}
	if (fe->events & EventRead || fe->events & EventWait2Read)
	{		
		if (fe->events & EventWait2Read && fe->watcherRCmsTimer !=  mhttp->cmsTimer2Read())
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		else if (fe->events & EventRead && mwatcherReadIO != fe->watcherReadIO)
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		return doRead(fe->events & EventWait2Read);
	}
	if (fe->events & EventErrot)
	{
		logs->error("%s [CHttpServer::handleEv] handlEv recv event error ***",
			mremoteAddr.c_str());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CHttpServer::stop(std::string reason)
{
	//可能会被调用两次,任务断开时,正常调用一次 reason 为空,
	//主动断开时,会调用,reason 是调用原因
	if (!reason.empty())
	{
		logs->info("%s [CHttpServer::stop] http %s stop with reason: %s ***",
			mremoteAddr.c_str(),murl.c_str(),reason.c_str());
	}
	else
	{
		logs->info("%s [CHttpServer::stop] http %s has been stop ",
			mremoteAddr.c_str(),murl.c_str(),reason.c_str());
		if (misAddConn)
		{
			down8upBytes();
			makeOneTaskupload(mHash,0,PACKET_CONN_DEL);
		}
	}
	misStop = true;
	return CMS_OK;
}

std::string CHttpServer::getUrl()
{
	return murl;
}

std::string CHttpServer::getRemoteIP()
{
	return mremoteIP;
}

void CHttpServer::setEVLoop(struct ev_loop *loop)
{
	mloop = loop;
}

struct ev_loop *CHttpServer::evLoop()
{
	return mloop;
}

struct ev_io *CHttpServer::evReadIO()
{
	if (mwatcherReadIO == NULL)
	{
		mwatcherReadIO = new (ev_io);
		ev_io_init(mwatcherReadIO, readEV, mrw->fd(), EV_READ);
		ev_io_start(mloop, mwatcherReadIO);
	}
	return mwatcherReadIO;
}

struct ev_io *CHttpServer::evWriteIO()
{
	if (mwatcherWriteIO == NULL)
	{
		mwatcherWriteIO = new (ev_io);
		ev_io_init(mwatcherWriteIO, writeEV, mrw->fd(), EV_WRITE);
		ev_io_start(mloop, mwatcherWriteIO);
	}
	return mwatcherWriteIO;
}

int CHttpServer::doRead(bool isTimeout)
{
	return mhttp->want2Read(isTimeout);
}

int CHttpServer::doWrite(bool isTimeout)
{	
	return mhttp->want2Write(isTimeout);
}

int CHttpServer::doDecode()
{
	int ret = CMS_OK;
	if (!misDecodeHeader)
	{
		murl = mhttp->httpRequest()->getUrl();
		misDecodeHeader = true;
		std::string strUpgrade = mhttp->httpRequest()->getHeader(HTTP_HEADER_UPGRADE);
		if (!strUpgrade.empty())
		{
			//web socket
			std::string strConnect = mhttp->httpRequest()->getHeader(HTTP_HEADER_CONNECTION);
			std::string strSecWebSocketVersion = mhttp->httpRequest()->getHeader(HTTP_HEADER_SEC_WEBSOCKET_VER);
			for ( string::iterator iterKey = strUpgrade.begin(); iterKey != strUpgrade.end(); iterKey++ )
			{
				if ( *iterKey <= 'Z' && *iterKey >= 'A' )
				{
					*iterKey += 32;
				}
			}
			for ( string::iterator iterKey = strConnect.begin(); iterKey != strConnect.end(); iterKey++ )
			{
				if ( *iterKey <= 'Z' && *iterKey >= 'A' )
				{
					*iterKey += 32;
				}
			}
			if (!(strConnect == HTTP_HEADER_UPGRADE && 
				strUpgrade == HTTP_HEADER_WEBSOCKET &&
				strSecWebSocketVersion == "13"))
			{
				logs->debug("***** %s [CHttpServer::doDecode] %s websocket http header error *****",
					mremoteAddr.c_str(),murl.c_str());
				return CMS_ERROR;
			}
			std::string strSecWebSocketKey = mhttp->httpRequest()->getHeader(HTTP_HEADER_SEC_WEBSOCKET_KEY);
			mwsOrigin = mhttp->httpRequest()->getHeader(HTTP_HEADER_ORIGIN);
			std::string strSecWebSocketProtocol = mhttp->httpRequest()->getHeader(HTTP_HEADER_SEC_WEBSOCKET_PROTOCOL);
			std::vector<std::string> swp;
			std::string delim = ",";
			split(strSecWebSocketProtocol,delim,swp);
			if (!swp.empty())
			{
				delim = " ";
				msecWebSocketProtocol = trim(swp[0],delim);
			}
			strSecWebSocketKey += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
			logs->debug("%s [CHttpServer::doDecode] %s websocket make key string %s",
				mremoteAddr.c_str(),murl.c_str(),strSecWebSocketKey.c_str());
			HASH hash = ::makeHash(strSecWebSocketKey.c_str(),strSecWebSocketKey.length());
			strSecWebSocketKey = hash2Char(hash.data);
			logs->debug("%s [CHttpServer::doDecode] %s websocket make string hash %s",
				mremoteAddr.c_str(),murl.c_str(),strSecWebSocketKey.c_str());
			std::string strHex;
			char szCode[21] = {0};
			for (int i = 0; i < (int)strSecWebSocketKey.length(); i++)
			{
				strHex.append(1,strSecWebSocketKey.at(i));
				if ((i+1) % 2 == 0)
				{
					long n = strtol(strHex.c_str(),NULL,16);
					szCode[i/2] = char(n);
					strHex = "";
				}
			}
			strHex = szCode;
			msecWebSocketAccept = getBase64Encode(strHex); //base64 可能不是标准的
			misWebSocket = true;
			mbinaryWriter = new BinaryWriter;
		}		
		logs->debug("%s [CHttpServer::doDecode] new httpserver request %s",
			mremoteAddr.c_str(),murl.c_str());
		ret = handle();
	}
	return ret;
}

int CHttpServer::handle()
{
	int ret = CMS_OK;
	do 
	{
		if (handleFlv(ret) != 0)
		{
			break;
		}
		if (handleQuery(ret) != 0)
		{
			break;
		}
		logs->warn("***** %s [CHttpServer::handleFlv] http %s unknow request *****",
			mremoteAddr.c_str(),murl.c_str());
		ret = CMS_ERROR;
	} while (0);
	return ret;
}

int	CHttpServer::handleFlv(int &ret)
{
	if (murl.find(".flv") != string::npos)
	{
		LinkUrl linkUrl;
		if (!parseUrl(murl,linkUrl))
		{
			logs->error("***** %s [CHttpServer::handleFlv] http %s parse url fail *****",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		if (isLegalIp(linkUrl.host.c_str()))
		{
			//302 地址
			murl = "http://";
			murl += linkUrl.app;
			murl += "/";
			murl += linkUrl.instanceName;
			logs->debug(">>> %s [CHttpServer::handleFlv] http 302 ip url %s ",
				mremoteAddr.c_str(),murl.c_str());
			if (!parseUrl(murl,linkUrl))
			{
				logs->error("*** %s [CHttpServer::handleFlv] http 302 url %s parse fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
		}
		misFlvRequest = true;
		mreferer = mhttp->httpRequest()->getHeader(HTTP_HEADER_REQ_REFERER);
		makeHash();
		tryCreateTask();

		//succ
		if (misWebSocket)
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_101,"Switching Protocols");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_UPGRADE,"websocket");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"Upgrade");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_ACCESS_CONTROL_ALLOW_ORIGIN,"*");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SEC_WEBSOCKET_ACCEPT,msecWebSocketAccept);
			if (!msecWebSocketProtocol.empty())
			{
				mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SEC_WEBSOCKET_PROTOCOL,msecWebSocketProtocol);
			}
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER,"cms server");
		}
		else
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"video/x-flv");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER,"cms server");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CACHE_CONTROL,"no-cache");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_PRAGMA,"no-cache");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_ACCESS_CONTROL_ALLOW_ORIGIN,"*");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"close");
		}
		std::string strRspHeader = mhttp->httpResponse()->readResponse();
		//flv header
		strRspHeader.append(1,0x46);
		strRspHeader.append(1,0x4C);
		strRspHeader.append(1,0x56);
		strRspHeader.append(1,0x01);
		strRspHeader.append(1,0x05);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x09);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x00);
		strRspHeader.append(1,0x00);
		ret = sendBefore(strRspHeader.c_str(),strRspHeader.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleFlv] http %s send header fail ***",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = doTransmission();
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleFlv] http %s doTransmission fail ***",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = CMS_OK;
		return 1;
	}
	return 0;
}

int CHttpServer::handleQuery(int &ret)
{
	if (murl.find("/query/info") != string::npos)
	{
		std::string strDump = CStatic::instance()->dump();
		char szLength[20] = {0};
		snprintf(szLength,sizeof(szLength),"%lu",strDump.length());
		//succ
		if (misWebSocket)
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_101,"Switching Protocols");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_UPGRADE,"websocket");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"Upgrade");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_ACCESS_CONTROL_ALLOW_ORIGIN,"*");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SEC_WEBSOCKET_ACCEPT,msecWebSocketAccept);
			if (!msecWebSocketProtocol.empty())
			{
				mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SEC_WEBSOCKET_PROTOCOL,msecWebSocketProtocol);
			}
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER,"cms server");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"text/json; charset=UTF-8");
		}
		else
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"text/json; charset=UTF-8");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER,"cms server");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"close");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
		}
		std::string strRspHeader = mhttp->httpResponse()->readResponse();
		strRspHeader += strDump;
		ret = sendBefore(strRspHeader.c_str(),strRspHeader.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleFlv] http %s send header fail ***",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = CMS_ERROR;
		return 1;
	}
	return 0;
}

int CHttpServer::doTransmission()
{
	int ret = 1;
	if (misFlvRequest)
	{
		ret = mflvTrans->doTransmission();
		if (ret == 1 && !misAddConn)
		{
			misAddConn = true;
			makeOneTaskupload(mHash,0,PACKET_CONN_ADD);
			down8upBytes();
		}		
	}
	return ret;
}

int CHttpServer::sendBefore(const char *data,int len)
{
	if (misWebSocket)
	{
		/*
		WebSocket数据帧结构如下图所示：
		0                   1                   2                   3
		0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
		+-+-+-+-+-------+-+-------------+-------------------------------+
		|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
		|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
		|N|V|V|V|       |S|             |   (if payload len==126/127)   |
		| |1|2|3|       |K|             |                               |
		+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
		|     Extended payload length continued, if payload len == 127  |
		+ - - - - - - - - - - - - - - - +-------------------------------+
		|                               |Masking-key, if MASK set to 1  |
		+-------------------------------+-------------------------------+
		| Masking-key (continued)       |          Payload Data         |
		+-------------------------------- - - - - - - - - - - - - - - - +
		:                     Payload Data continued ...                :
		+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
		|                     Payload Data continued ...                |
		+---------------------------------------------------------------+
		*/
		*mbinaryWriter << (char)(0x02 | (0x01 << 7));
		if (len < 126)
		{
			*mbinaryWriter << (char)(len & 0x7F);
		} 
		else if (len < 65536)
		{
			*mbinaryWriter << (char)(126);
			*mbinaryWriter << (char)(len >> 8);
			*mbinaryWriter << (char)(len);
		} 
		else 
		{
			*mbinaryWriter << (char)(127);
			int64 dl = int64(len);
			*mbinaryWriter << (char)(dl >> 56);
			*mbinaryWriter << (char)(dl >> 48);
			*mbinaryWriter << (char)(dl >> 40);
			*mbinaryWriter << (char)(dl >> 32);
			*mbinaryWriter << (char)(dl >> 24);
			*mbinaryWriter << (char)(dl >> 16);
			*mbinaryWriter << (char)(dl >> 8);
			*mbinaryWriter << (char)(dl);
		}
		int n = mbinaryWriter->getLength();
		if (mhttp->write(mbinaryWriter->getData(),n) == CMS_ERROR)
		{
			return CMS_ERROR;
		}
		mbinaryWriter->reset();
		return CMS_OK;
	}
	return mhttp->write(data,len);
}

void CHttpServer::makeHash()
{
	string hashUrl = readHashUrl(murl);
	CSHA1 sha;
	sha.write(hashUrl.c_str(), hashUrl.length());
	string strHash = sha.read();
	mHash = HASH((char *)strHash.c_str());
	mstrHash = hash2Char(mHash.data);
	mHashIdx = CFlvPool::instance()->hashIdx(mHash);
	logs->debug("%s [CHttpServer::makeHash] hash url %s,hash=%s",
		mremoteAddr.c_str(),hashUrl.c_str(),mstrHash.c_str());
	mflvTrans->setHash(mHashIdx,mHash);
}

void CHttpServer::tryCreateTask()
{
	if (!CTaskMgr::instance()->pullTaskIsExist(mHash))
	{
		CTaskMgr::instance()->createTask(murl,"",murl,mreferer,CREATE_ACT_PULL,false,false);
	}
}

void CHttpServer::down8upBytes()
{
	if (misFlvRequest)
	{
		unsigned long tt = getTickCount();
		if (tt - mspeedTick > 1000)
		{
			mspeedTick = tt;
			int32 bytes = mrdBuff->readBytesNum();
			if (bytes > 0)
			{
				makeOneTaskDownload(mHash,bytes,false);
			}
			bytes = mwrBuff->writeBytesNum();
			if (bytes > 0)
			{
				makeOneTaskupload(mHash,bytes,PACKET_CONN_DATA);
			}
		}
	}
}
