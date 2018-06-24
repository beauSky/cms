/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: ���û������/kisslovecsh@foxmail.com

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
#include <net/cms_net_mgr.h>
#include <ts/cms_hls_mgr.h>
#include <app/cms_app_info.h>
#include <regex.h>

std::string gCrossDomainRsp = "HTTP/1.1 200 OK\r\nServer: quick rtmp\r\nConnection: keep-alive\r\nContent-Length: 189\r\nContent-Type: text/xml\r\r\n<?xml version=\"1.0\"?><!DOCTYPE cross-domain-policy SYSTEM \"http://www.adobe.com/xml/dtds/cross-domain-policy.dtd\"><cross-domain-policy><allow-access-from domain=\"*\" /></cross-domain-policy>";

CHttpServer::CHttpServer(CReaderWriter *rw,bool isTls)
{
	char szaddr[23] = {0};
	rw->remoteAddr(szaddr,sizeof(szaddr));
	mremoteAddr = szaddr;
	size_t pos = mremoteAddr.find(":");
	if (pos == string::npos)
	{
		mremoteIP = mremoteAddr;
	}
	else
	{
		mremoteIP = mremoteAddr.substr(0,pos);
	}
	memset(szaddr,0,sizeof(szaddr));
	rw->localAddr(szaddr,sizeof(szaddr));
	mlocalAddr = szaddr;
	pos = mlocalAddr.find(":");
	if (pos == string::npos)
	{
		mlocalIP = mlocalAddr;
	}
	else
	{
		mlocalIP = mlocalAddr.substr(0,pos);
	}

	mrdBuff = new CBufferReader(rw,DEFAULT_BUFFER_SIZE);
	assert(mrdBuff);
	mwrBuff = new CBufferWriter(rw,DEFAULT_BUFFER_SIZE);
	assert(mwrBuff);
	mrw = rw;
	mhttp = new CHttp(this,mrdBuff,mwrBuff,rw,mremoteAddr,false,isTls);
	mwatcherReadIO = NULL;
	mwatcherWriteIO = NULL;
	mllIdx = 0;
	misDecodeHeader = false;
	misWebSocket = false;
	mflvTrans = new CFlvTransmission(mhttp);
	mbinaryWriter = NULL;
	misAddConn = false;
	misFlvRequest = false;
	misM3U8TSRequest = false;
	misStop = false;

	mspeedTick = 0;
	mtimeoutTick = getTimeUnix();

	//�ٶ�ͳ��
	mxSecdownBytes = 0;
	mxSecUpBytes = 0;
	mxSecTick = 0;
}

CHttpServer::~CHttpServer()
{
	logs->debug("######### %s [CHttpServer::~CHttpServer] http enter ",
		mremoteAddr.c_str());
	if (mwatcherReadIO)
	{
		if (mrw->netType() == NetTcp || (mrw->netType() == NetUdp && mrw->fd() > 0))//udp ������
		{
			CNetMgr::instance()->cneStop(mwatcherReadIO);
		}
		freeCmsNetEv(mwatcherReadIO);
		logs->debug("######### %s [CHttpServer::~CHttpServer] stop read io ",
			mremoteAddr.c_str());
	}
	if (mwatcherWriteIO)
	{
		if (mrw->netType() == NetTcp || (mrw->netType() == NetUdp && mrw->fd() > 0))//udp ������
		{
			CNetMgr::instance()->cneStop(mwatcherWriteIO);
		}
		freeCmsNetEv(mwatcherWriteIO);
		logs->debug("######### %s [CHttpServer::~CHttpServer] stop write io ",
			mremoteAddr.c_str());
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
	if (mrw->netType() == NetTcp)//udp ������
	{
		//udp ������udpģ��������� ����ҪҲ�������ⲿ�ͷ�
		delete mrw;
	}
}

void CHttpServer::reset()
{
	misDecodeHeader = false;
	misFlvRequest = false;;
	misM3U8TSRequest = false;
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
			//Ӧ���Ǿɵ�socket�ŵ���Ϣ
			return CMS_OK;
		}
		else if (fe->events & EventWrite && mwatcherWriteIO != fe->watcherWriteIO)
		{
			//Ӧ���Ǿɵ�socket�ŵ���Ϣ
			return CMS_OK;
		}
		return doWrite(fe->events & EventWait2Write);
	}
	if (fe->events & EventRead || fe->events & EventWait2Read)
	{		
		if (fe->events & EventWait2Read && fe->watcherRCmsTimer !=  mhttp->cmsTimer2Read())
		{
			//Ӧ���Ǿɵ�socket�ŵ���Ϣ
			return CMS_OK;
		}
		else if (fe->events & EventRead && mwatcherReadIO != fe->watcherReadIO)
		{
			//Ӧ���Ǿɵ�socket�ŵ���Ϣ
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
	//���ܻᱻ��������,����Ͽ�ʱ,��������һ�� reason Ϊ��,
	//�����Ͽ�ʱ,�����,reason �ǵ���ԭ��
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

cms_net_ev *CHttpServer::evReadIO(cms_net_ev *ev)
{
	if (mwatcherReadIO == NULL)
	{
		if (ev != NULL)
		{
			//�Զ����socket ������
			atomicInc(ev);		//��������1
			mwatcherReadIO = ev;
		}
		else
		{			
			mwatcherReadIO = mallcoCmsNetEv();
			initCmsNetEv(mwatcherReadIO,readEV,mrw->fd(),EventRead);
			CNetMgr::instance()->cneStart(mwatcherReadIO);
		}
	}
	return mwatcherReadIO;
}

cms_net_ev *CHttpServer::evWriteIO(cms_net_ev *ev)
{
	if (mwatcherWriteIO == NULL)
	{
		if (ev != NULL)
		{
			//�Զ����socket ������
			atomicInc(ev);		//��������1
			mwatcherWriteIO = ev;
		}
		else
		{			
			mwatcherWriteIO = mallcoCmsNetEv();
			initCmsNetEv(mwatcherWriteIO,writeEV,mrw->fd(),EventWrite);
			CNetMgr::instance()->cneStart(mwatcherWriteIO);
		}
	}
	return mwatcherWriteIO;
}

int CHttpServer::doRead(bool isTimeout)
{
	int64 tn = getTimeUnix();
	if (tn - mtimeoutTick > 30)
	{
		logs->error("%s [CHttpServer::doRead] http %s is timeout ***",
			mremoteAddr.c_str(),murl.c_str());
		return CMS_ERROR;
	}
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
				strSecWebSocketVersion == HTTP_HEADER_WEBSOCKET_VERSION_NUM))
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
			msecWebSocketAccept = getBase64Encode(strHex); //base64 ��׼��
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
		if (handleCrossDomain(ret) != 0)
		{
			break;
		}
		if (handleFlv(ret) != 0)
		{
			break;
		}
		if (handleQuery(ret) != 0)
		{
			break;
		}

		if (handleM3U8(ret) != 0)
		{
			break;
		}
		if (handleTS(ret) != 0)
		{
			break;
		}
		//Ĭ�ϲ���http-flv
		if (handleFlv(ret, true) != 0)
		{
			break;
		}
		logs->warn("***** %s [CHttpServer::handleFlv] http %s unknow request *****",
			mremoteAddr.c_str(),murl.c_str());
		ret = CMS_ERROR;
	} while (0);
	return ret;
}

int CHttpServer::handleCrossDomain(int &ret)
{
	if (murl.find("crossdomain.xml") != string::npos)
	{
		ret = sendBefore(gCrossDomainRsp.c_str(),gCrossDomainRsp.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleCrossDomain] http %s send header fail ***",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = CMS_OK;
		return 1;
	}
	return 0;
}

int	CHttpServer::handleFlv(int &ret, bool isDefault/* = false*/)
{
	if (murl.find(".flv") != string::npos || isDefault)
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
			if (linkUrl.app.empty() || linkUrl.instanceName.empty())
			{
				logs->error("*** %s [CHttpServer::handleFlv] http 302 url %s app or instance name should be empty ***",
					mremoteAddr.c_str(), murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
			//302 ��ַ
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
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
		}
		else
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"video/x-flv");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CACHE_CONTROL,"no-cache");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_PRAGMA,"no-cache");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_ACCESS_CONTROL_ALLOW_ORIGIN,"*");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"close");
		}
		std::string strRspHeader = mhttp->httpResponse()->readResponse();
		ret = writeRspHttpHeader(strRspHeader.c_str(), strRspHeader.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleFlv] http %s send header fail ***",
				mremoteAddr.c_str(), murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		//flv header
		std::string strFlvHeader;
		strFlvHeader.append(1,0x46);
		strFlvHeader.append(1,0x4C);
		strFlvHeader.append(1,0x56);
		strFlvHeader.append(1,0x01);

		/*if (mhttp->httpRequest()->getHttpParam("only-audio") == "1")
		{
			strFlvHeader.append(1, 0x04);
		}
		else if (mhttp->httpRequest()->getHttpParam("only-video") == "1")
		{
			strFlvHeader.append(1, 0x01);
		}
		else 
		{
			strFlvHeader.append(1, 0x05);
		}*/
		strFlvHeader.append(1, 0x05);

		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x09);
		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x00);
		strFlvHeader.append(1,0x00);
		ret = sendBefore(strFlvHeader.c_str(),strFlvHeader.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleFlv] http %s send body fail ***",
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
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"text/json; charset=UTF-8");
		}
		else
		{
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"text/json; charset=UTF-8");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"close");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
		}
		std::string strRspHeader = mhttp->httpResponse()->readResponse();
		ret = writeRspHttpHeader(strRspHeader.c_str(), strRspHeader.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleQuery] http %s send header fail ***",
				mremoteAddr.c_str(), murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = sendBefore(strDump.c_str(), strDump.length());
		if (ret < 0)
		{
			logs->error("*** %s [CHttpServer::handleQuery] http %s send body fail ***",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		ret = CMS_OK;
		return 1;
	}
	return 0;
}

int  CHttpServer::handleM3U8(int &ret)
{
	static char pattern[] = "[0-9A-Za-z_]+/online.m3u8";
	size_t nmatch = 1;
	regmatch_t pm[1];
	regex_t reg;
	regcomp(&reg,pattern,REG_EXTENDED|REG_NOSUB);
	int r = regexec(&reg,murl.c_str(),nmatch,pm,REG_NOTBOL);
	regfree(&reg);
	if (r != REG_NOMATCH)
	{
		logs->debug(" %s [CHttpServer::handleM3U8] http %s is m3u8 request.",
			mremoteAddr.c_str(),murl.c_str());
		misM3U8TSRequest = true;
		LinkUrl linkUrl;
		if (!parseUrl(murl,linkUrl))
		{
			logs->error("***** %s [CHttpServer::handleM3U8] http %s parse url fail *****",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		if (isLegalIp(linkUrl.host.c_str()))
		{
			//302 ��ַ
			murl = "http://";
			murl += linkUrl.app;
			murl += "/";
			murl += linkUrl.instanceName;
			logs->debug(">>> %s [CHttpServer::handleM3U8] http 302 ip url %s ",
				mremoteAddr.c_str(),murl.c_str());
			if (!parseUrl(murl,linkUrl))
			{
				logs->error("*** %s [CHttpServer::handleM3U8] http 302 url %s parse fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
		}
		mreferer = mhttp->httpRequest()->getHeader(HTTP_HEADER_REQ_REFERER);

		std::string url = murl;
		size_t pos = url.rfind("/");
		if (pos != std::string::npos)
		{
			url = url.substr(0,pos);
		}
		makeHash(url);
		std::string outData;
		int64 outTT;
		int ret = CMissionMgr::instance()->readM3U8(mHashIdx,mHash,murl,mlocalAddr,outData,outTT);
		if (ret > 0)
		{
			logs->debug(">>> %s [CHttpServer::handleM3U8] %s ,local addr %s,m3u8\n %s ",
				mremoteAddr.c_str(),murl.c_str(),mlocalAddr.c_str(),outData.c_str());

			char szLength[20] = {0};
			snprintf(szLength,sizeof(szLength),"%lu",outData.length());
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"application/vnd.apple.mpegurl");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"keep-alive");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);

			std::string strRspHeader = mhttp->httpResponse()->readResponse();
			ret = writeRspHttpHeader(strRspHeader.c_str(), strRspHeader.length());
			if (ret < 0)
			{
				logs->error("*** %s [CHttpServer::handleM3U8] http %s send header fail ***",
					mremoteAddr.c_str(), murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
			ret = sendBefore(outData.c_str(), outData.length());
			if (ret < 0)
			{
				logs->error("*** %s [CHttpServer::handleM3U8] http %s send body fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
			ret = CMS_OK;
		}
		else
		{
			char szLength[20] = {0};
			snprintf(szLength,sizeof(szLength),"%d",0);
			mhttp->httpResponse()->setStatus(HTTP_CODE_404,"Not Found");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"keep-alive");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
			std::string strRspHeader = mhttp->httpResponse()->readResponse();
			ret = writeRspHttpHeader(strRspHeader.c_str(),strRspHeader.length());
			if (ret < 0)
			{
				logs->error("*** %s [CHttpServer::handleM3U8] http %s send header fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
			ret = CMS_OK;
		}
		mtimeoutTick = getTimeUnix();
		return 1;
	}
	return 0;
}

int  CHttpServer::handleTS(int &ret)
{
	static char pattern[] = "[0-9A-Za-z]+/[0-9]+.ts";
	size_t nmatch = 1;
	regmatch_t pm[1];
	regex_t reg;
	regcomp(&reg,pattern,REG_EXTENDED|REG_NOSUB);
	int r = regexec(&reg,murl.c_str(),nmatch,pm,REG_NOTBOL);
	regfree(&reg);
	if (r != REG_NOMATCH)
	{
		logs->debug(" %s [CHttpServer::handleTS] http %s is ts request.",
			mremoteAddr.c_str(),murl.c_str());
		misM3U8TSRequest = true;
		LinkUrl linkUrl;
		if (!parseUrl(murl,linkUrl))
		{
			logs->error("***** %s [CHttpServer::handleTS] http %s parse url fail *****",
				mremoteAddr.c_str(),murl.c_str());
			ret = CMS_ERROR;
			return CMS_ERROR;
		}
		if (isLegalIp(linkUrl.host.c_str()))
		{
			//302 ��ַ
			murl = "http://";
			murl += linkUrl.app;
			murl += "/";
			murl += linkUrl.instanceName;
			logs->debug(">>> %s [CHttpServer::handleTS] http 302 ip url %s ",
				mremoteAddr.c_str(),murl.c_str());
			if (!parseUrl(murl,linkUrl))
			{
				logs->error("*** %s [CHttpServer::handleTS] http 302 url %s parse fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
		}
		mreferer = mhttp->httpRequest()->getHeader(HTTP_HEADER_REQ_REFERER);

		std::string url = murl;
		size_t pos = url.rfind("/");
		if (pos != std::string::npos)
		{
			url = url.substr(0,pos);
		}
		makeHash(url);
		int64 outTT;
		SSlice *ss;
		int ret = CMissionMgr::instance()->readTS(mHashIdx,mHash,murl,mlocalAddr,&ss,outTT);
		if (ret > 0)
		{
			char szLength[20] = {0};
			snprintf(szLength,sizeof(szLength),"%d",ss->msliceLen);
			logs->debug(" %s [CHttpServer::handleTS] http %s ts size %d",
				mremoteAddr.c_str(),murl.c_str(),ss->msliceLen);
			mhttp->httpResponse()->setStatus(HTTP_CODE_200,"OK");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_TYPE,"video/mp2t");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"keep-alive");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
			std::string strRspHeader = mhttp->httpResponse()->readResponse();
			ret = writeRspHttpHeader(strRspHeader.c_str(),strRspHeader.length());
			if (ret < 0)
			{
				logs->error("*** %s [CHttpServer::handleTS] http %s send header fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}

			bool isError = false;
			TsChunkArray *tca = NULL;
			std::vector<TsChunkArray *>::iterator it = ss->marray.begin();
			for (; it != ss->marray.end() && !isError; it++)
			{
				tca = *it;
				int iBegin = beginChunk(tca);
				int iEnd = endChunk(tca);
				if (iBegin != -1 && iEnd != -1)
				{
					for (; iBegin < iEnd; iBegin++)
					{
						TsChunk *tc = NULL;
						getChunk(tca,iBegin,&tc);
						if (tc)
						{
							assert(tc->muse%TS_CHUNK_SIZE==0);
							ret = sendBefore(tc->mdata,tc->muse);			
							if (ret < 0)
							{
								isError = true;
								logs->error("*** %s [CHttpServer::handleTS] http %s send header fail ***",
									mremoteAddr.c_str(),murl.c_str());
								ret = CMS_ERROR;
							}
						}
						else
						{
							assert(0);
						}
					}
				}
			}
			atomicDec(ss);
			if (!isError)
			{
				ret = CMS_OK;
			}
		}else
		{
			char szLength[20] = {0};
			snprintf(szLength,sizeof(szLength),"%d",0);
			mhttp->httpResponse()->setStatus(HTTP_CODE_404,"Not Found");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_SERVER, APP_NAME);
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONNECTION,"keep-alive");
			mhttp->httpResponse()->setHeader(HTTP_HEADER_RSP_CONTENT_LENGTH,szLength);
			std::string strRspHeader = mhttp->httpResponse()->readResponse();
			ret = writeRspHttpHeader(strRspHeader.c_str(),strRspHeader.length());
			if (ret < 0)
			{
				logs->error("*** %s [CHttpServer::handleTS] http %s send header fail ***",
					mremoteAddr.c_str(),murl.c_str());
				ret = CMS_ERROR;
				return CMS_ERROR;
			}
			ret = CMS_OK;
		}
		mtimeoutTick = getTimeUnix();
		return 1;
	}
	return 0;
}

int CHttpServer::doTransmission()
{
	int ret = 1;
	if (misFlvRequest)
	{
		bool isSendData = false;
		ret = mflvTrans->doTransmission(isSendData);
		if (!misAddConn && (ret == 1 || ret == 0))
		{
			misAddConn = true;
			makeOneTaskupload(mHash,0,PACKET_CONN_ADD);
			down8upBytes();
		}
		if (isSendData)
		{
			mtimeoutTick = getTimeUnix();
		}
	}
	return ret;
}

int CHttpServer::writeRspHttpHeader(const char *data, int len)
{
	return mhttp->write(data, len);
}

int CHttpServer::sendBefore(const char *data,int len)
{
	if (misWebSocket)
	{
		/*
		WebSocket����֡�ṹ����ͼ��ʾ��
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
	}
	return mhttp->write(data,len);
}

bool CHttpServer::isFinish()
{
	if (misM3U8TSRequest)
	{
		return true;
	}
	return false;
}

bool CHttpServer::isWebsocket()
{
	return misWebSocket;
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

void CHttpServer::makeHash(std::string url)
{
	string hashUrl = readHashUrl(url);
	CSHA1 sha;
	sha.write(hashUrl.c_str(), hashUrl.length());
	string strHash = sha.read();
	mHash = HASH((char *)strHash.c_str());
	mstrHash = hash2Char(mHash.data);
	mHashIdx = CFlvPool::instance()->hashIdx(mHash);
	logs->debug("%s [CHttpServer::makeHash] 1 hash url %s,hash=%s",
		mremoteAddr.c_str(),hashUrl.c_str(),mstrHash.c_str());
}

void CHttpServer::tryCreateTask()
{
	if (!CTaskMgr::instance()->pullTaskIsExist(mHash))
	{
		CTaskMgr::instance()->createTask(mHash,murl,"",murl,mreferer,CREATE_ACT_PULL,false,false);
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

			mxSecdownBytes += bytes;

			bytes = mwrBuff->writeBytesNum();
			if (bytes > 0)
			{
				makeOneTaskupload(mHash,bytes,PACKET_CONN_DATA);
			}

			mxSecUpBytes += bytes;
			mxSecTick++;
			if (((mxSecTick+(0x0F-(CMS_SPEED_DURATION>=0x0F?10:CMS_SPEED_DURATION)+1)) & 0x0F) == 0)
			{
				logs->debug("%s [CHttpServer::down8upBytes] http %s download speed %s,upload speed %s",
					mremoteAddr.c_str(),murl.c_str(),
					parseSpeed8Mem(mxSecdownBytes/mxSecTick,true).c_str(),
					parseSpeed8Mem(mxSecUpBytes/mxSecTick,true).c_str());
				mxSecTick = 0;
				mxSecdownBytes = 0;
				mxSecUpBytes = 0;
			}
		}
	}
}

CReaderWriter *CHttpServer::rwConn()
{
	return mrw;
}
