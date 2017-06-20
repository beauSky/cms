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
#include <protocol/cms_http.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <conn/cms_conn_var.h>
#include <ev/cms_ev.h>
#include <sstream>
#include <assert.h>

#define mapStrStrIterator std::map<std::string,std::string>::iterator 

bool parseHttpHeader(const char *buf,int len,map<string,string> &mapKeyValue)
{
	bool bSucc = true;
	char *pPosHeaderEnd = NULL;
	mapKeyValue.clear();
	pPosHeaderEnd = (char *)strstr( buf, "\r\n\r\n" );
	if ( NULL != pPosHeaderEnd )
	{
		pPosHeaderEnd += 4;
		int lenHeader = pPosHeaderEnd - buf;
		if ( lenHeader > 11 )
		{
			string strKey;
			string strValue;
			char *pPosEnd = NULL;
			char *pPosEndTrim = NULL;
			char *pPosStart = NULL;
			map<string,string >::iterator iterHeaderField;
			for ( pPosEnd = (char *)buf, pPosStart =(char *) buf; pPosEnd <= pPosHeaderEnd; pPosEnd++ )
			{
				// get key
				if ( *pPosEnd == ':' && strKey.empty() )
				{
					// trim left
					pPosEndTrim = pPosEnd;
					while ( ' ' == *pPosStart && pPosStart < pPosEndTrim )
					{
						pPosStart++;
					}
					// trim right
					while ( ' ' == *pPosEndTrim && pPosEndTrim > pPosStart )
					{
						pPosEndTrim--;
					}
					strKey.append( pPosStart, pPosEndTrim );
					pPosEnd+=2;
					pPosStart = pPosEnd;
				}
				// get value & set attribute
				if ( *pPosEnd == '\r' && *(pPosEnd+1) == '\n' )
				{
					// trim left
					pPosEndTrim = pPosEnd;
					while ( ' ' == *pPosStart && pPosStart < pPosEndTrim )
					{
						pPosStart++;
					}
					// trim right
					while ( ' ' == *pPosEndTrim && pPosEndTrim > pPosStart )
					{
						pPosEndTrim--;
					}

					strValue.append( pPosStart, pPosEndTrim );
					pPosEnd+=2;
					pPosStart = pPosEnd;
					if ( !strKey.empty() && !strValue.empty() )
					{
						// key lower case
						for ( string::iterator iterKey = strKey.begin(); iterKey != strKey.end(); iterKey++ )
						{
							if ( *iterKey <= 'Z' && *iterKey >= 'A' )
							{
								*iterKey += 32;
							}
						}

						mapKeyValue[ strKey ] = strValue;
					}
					strKey.clear();
					strValue.clear();
				}
			}
		}
		else
		{
			bSucc = false;
		}
	}
	else
	{
		bSucc = false;
	}
	return bSucc;
}

int splitStr(const string& str, vector<string>& ret_, char ch)
{
	if (str.empty())
	{
		return 0;
	}

	string tmp = str;
	string::size_type pos_begin = tmp.find(ch);
	string strVal;
	while (pos_begin != string::npos)
	{
		strVal = tmp.substr(0,pos_begin);
		ret_.push_back(strVal);
		tmp = tmp.substr(++pos_begin);
		pos_begin = tmp.find(ch);
	}
	ret_.push_back(tmp);
	return 0;
}

bool parseHttpParam(char *path,map<string,string> &mapParamValue)
{
	char *pPos = strstr(path,"?");
	if (pPos == NULL)
	{
		return true;
	}
	string strParam = pPos+1;
	*pPos = '\0';
	vector<string> vctVal;
	splitStr(strParam,vctVal,'&');
	vector<string>::iterator it = vctVal.begin();
	for ( ; it != vctVal.end(); ++it)
	{
		string p,v;
		string::size_type pos = it->find("=");
		if (pos)
		{
			p = it->substr(0,pos);
			v = it->substr(pos+1);
		}
		else
		{
			p = *it;
		}		
		mapParamValue.insert(make_pair(p,v));
	}
	return true;
}

Request::Request()
{
	mmethod = "GET";
}

Request::~Request()
{

}

void Request::setMethod(std::string method)
{
	mmethod = method;
}

bool Request::setUrl(std::string url)
{
	murl = url;
	if (!parseUrl(url,mlinkUrl))
	{
		logs->error("***** [Request::setUrl] parse url %s fail *****",murl.c_str());
		return false;
	}
	setHeader(HTTP_HEADER_HOST,mlinkUrl.addr);
	return true;
}

std::string Request::getUrl()
{
	return murl;
}

void Request::setRefer(std::string refer)
{
	mreferUrl = refer;
}
	
void Request::setHeader(std::string key,std::string value)
{
	mmapHeader[key] = value;
}

std::string	Request::getHeader(std::string key)
{
	mapStrStrIterator it = mmapHeader.find(key);
	if (it != mmapHeader.end())
	{
		return it->second;
	}
	return "";
}

void Request::clearHeader()
{
	mmapHeader.clear();
}

std::string Request::readRequest()
{
	std::ostringstream strStream;
	strStream << mmethod << " " << mlinkUrl.uri << " " << HTTP_VERSION << "\r\n";
	for (mapStrStrIterator it = mmapHeader.begin(); it != mmapHeader.end(); ++it)
	{
		strStream << it->first << ": " << it->second << "\r\n";
	}
	strStream << "\r\n";
	return strStream.str();
}

void Request::setRemoteAddr(std::string addr)
{
	mremoteAddr = addr;
}

std::string Request::getRemoteAddr()
{
	return mremoteAddr;
}

bool Request::parseHeader(const char *header,int len)
{
	char cmd[10]="", path[9096+4]="", version[10]="";
	sscanf(header, "%9s %9096s %9s", cmd, path, version);
	logs->debug("+++ %s [Request::parseHeader] cmd[ %s ],path[ %s ],version[ %s ] +++",
		mremoteAddr.c_str(), cmd, path, version);
	if (!parseHttpHeader(header,len,mmapHeader))
	{
		logs->error("***** %s [Request::parseHeader] parseHttpHeader fail *****",
			mremoteAddr.c_str());
		return false;
	}
	if (!parseHttpParam(path,mmapParam))
	{
		logs->error("***** %s [Request::parseHeader] parseHttpParam fail *****",
			mremoteAddr.c_str());
		return false;
	}
	mmethod = cmd;
	mapStrStrIterator it = mmapHeader.find(HTTP_HEADER_HOST);
	if (it == mmapHeader.end())
	{
		return false;
	}
	murl = "http://" + it->second;
	if (strlen(path) == 0)
	{
		murl += "/";
	}
	else 
	{
		if (path[0] != '/')
		{
			murl += "/";
		}
		murl += path;
	}
	logs->debug("##### %s [Request::parseHeader] request url %s ",
		mremoteAddr.c_str(),murl.c_str());
	return true;
}

void Request::reset()
{
	mremoteAddr.clear();
	mmethod.clear();
	resetLinkUrl(&mlinkUrl);
	murl.clear();
	mreferUrl.clear();
	mmapHeader.clear();
	mmapParam.clear();
	mhost.clear();
}

Response::Response()
{

}

Response::~Response()
{

}

std::string	Response::getHeader(std::string key)
{
	mapStrStrIterator it = mmapHeader.find(key);
	if (it != mmapHeader.end())
	{
		return it->second;
	}
	return "";
}

void Response::setHeader(std::string key,std::string value)
{
	mmapHeader[key] = value;
}

void Response::setUrl(std::string url)
{
	moriUrl = url;
}

void Response::clearHeader()
{
	mmapHeader.clear();
}

std::string Response::readResponse()
{
	std::ostringstream strStream;
	strStream << HTTP_VERSION << " " << mstatusCode << " " << mstatus << "\r\n";
	for (mapStrStrIterator it = mmapHeader.begin(); it != mmapHeader.end(); ++it)
	{
		strStream << it->first << ": " << it->second << "\r\n";
	}
	strStream << "\r\n";
	return strStream.str();
}

int Response::getStatusCode()
{
	return mstatusCode;
}

void Response::setStatus(int statusCode,std::string statsu)
{
	mstatusCode = statusCode;
	mstatus = statsu;
}

void Response::setRemoteAddr(std::string addr)
{
	mremoteAddr = addr;
}

std::string Response::getResponse()
{
	return moriRsp;
}

bool Response::parseHeader(const char *header,int len)
{
	moriRsp.append(header,len);
	mmapHeader.clear();
	char version[10]={0}, reason[128+4]={0}, status[10]={0};
	sscanf(header, "%9s %9s %128s", version, status, reason);
	logs->debug("+++ %s [Response::parseHeader] %s version[ %s ],reason[ %s ],status[ %s ] +++",
		mremoteAddr.c_str(),moriUrl.c_str(), version, reason, status);
	if (!parseHttpHeader(header,len,mmapHeader))
	{
		logs->error("***** %s [Response::parseHeader] %s parseHttpHeader fail *****",
			mremoteAddr.c_str(),moriUrl.c_str());
		return false;
	}
	mstatusCode = atoi(status);
	mstatus = reason;
	if (mstatusCode == HTTP_CODE_301 || mstatusCode == HTTP_CODE_302 || mstatusCode == HTTP_CODE_303)
	{
		mapStrStrIterator it = mmapHeader.find(HTTP_HEADER_LOCATION);
		if (it != mmapHeader.end())
		{
			if (it->second.find("http://") == string::npos)
			{
				LinkUrl linkUrl;
				if (!parseUrl(moriUrl,linkUrl))
				{
					logs->error("***** %s [Response::parseHeader] %s parseUrl fail *****",
						mremoteAddr.c_str(),moriUrl.c_str());
					return false;
				}
				std::string url302 = "http://" + linkUrl.addr + linkUrl.uri;
				logs->debug("+++ %s [Response::parseHeader] %s response 302 url %s +++",
					mremoteAddr.c_str(),moriUrl.c_str(), url302.c_str());
				mmapHeader["location"] = url302;
			}
		}
	}
	return true;
}

void Response::reset()
{
	moriUrl.clear();
	mremoteAddr.clear();
	mstatus.clear();
	mstatusCode = 0;
	mproto.clear();
	moriRsp.clear();
	mmapHeader.clear();
}

CHttp::CHttp(Conn *super,CBufferReader *rd,
	  CBufferWriter *wr,CReaderWriter *rw,std::string remoteAddr,
	  bool isClient,bool isTls)
{
	msuper = super;
	mremoteAddr = remoteAddr;
	mrdBuff = rd;
	mwrBuff = wr;
	mrw = rw;
	misClient = isClient;
	misTls = isTls;
	mcmsReadTimeout = NULL;
	mcmsWriteTimeout = NULL;
	mcmsReadTimeOutDo = 0;
	mcmsWriteTimeOutDo = 0;
	mrequest = NULL;
	mresponse = NULL;
	mssl = NULL;
	misReadHeader = false;
	misWriteHeader = false;
	mheaderEnd = 0;
	mbinaryWriter = new BinaryWriter;
	msendRequestLen = 0;
	misReadReuqest = false;
	misChunked = false;
	mchunkedLen = 0;
	misReadChunkedLen = false;
	mbyteReadWrite = NULL;
	mchunkedReadrRN = 0;
}

CHttp::~CHttp()
{
	if (mcmsWriteTimeout)
	{
		//��ĵط����ܻ���ʹ�ã�����ֱ��delete
		freeCmsTimer(mcmsWriteTimeout);
	}
	if (mcmsReadTimeout)
	{
		//��ĵط����ܻ���ʹ�ã�����ֱ��delete
		freeCmsTimer(mcmsReadTimeout);
	}
	if (mrequest)
	{
		delete mrequest;
	}
	if (mresponse)
	{
		delete mresponse;
	}
	if (mssl)
	{
		delete mssl;
	}
	if (mbinaryWriter)
	{
		delete mbinaryWriter;
	}
	if (mbyteReadWrite)
	{
		delete mbyteReadWrite;
	}
}

bool CHttp::run()
{
	if (misTls)
	{
		mssl = new CSSL(mrw->fd(),mremoteAddr,misClient);
		if (!mssl->run())
		{
			logs->error("***** %s [CHttp::run] %s cssl run fail *****", 
				mremoteAddr.c_str(),moriUrl.c_str());
			return false;
		}
	}
	else
	{

	}
	doReadTimeout();
	return true;
}

int CHttp::want2Read(bool isTimeout)
{	
	if (isTimeout)
	{
		assert(mcmsReadTimeOutDo==1);
		mcmsReadTimeOutDo--;
		doReadTimeout();
		//�������´�������
		return CMS_OK;
	}
	int ret = 0;
	for (;!misReadHeader;)
	{		
		char *p = NULL;
		char ch;
		int  len = 1;
		if (misTls)
		{
			ret = mssl->read(&p,len);
			if (ret < 0)
			{
				ret = -1;
				break;
			}
			else if (ret == 0)
			{
				ret = 0;
				break;
			}
			ch = *p;
		}
		else
		{
			if ( mrdBuff->size() < 1 && mrdBuff->grow(1) == CMS_ERROR)
			{
				logs->error("%s [CHttp::want2Read] %s http header read one byte fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),moriUrl.c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
				ret = -1;
				break;
			}
			if (mrdBuff->size() < 1)
			{
				ret = 0;
				break;
			}			
			ch = mrdBuff->readByte();
		}
		mstrHeader.append(1,ch);
		if (ch == '\r')
		{
			if (mheaderEnd % 2 == 0)
			{
				mheaderEnd++;
			}
			else
			{
				logs->warn("##### %s [CHttp::want2Read] %s 1 read header unexpect #####", 
					mremoteAddr.c_str(),moriUrl.c_str());
				mheaderEnd = 0;
			}
		}
		else if (ch == '\n')
		{
			if (mheaderEnd % 2 == 1)
			{
				mheaderEnd++;
			}
			else
			{
				logs->warn("##### %s [CHttp::want2Read] %s 2 read header unexpect #####", 
					mremoteAddr.c_str(),moriUrl.c_str());
				mheaderEnd = 0;
			}
		}
		else
		{
			mheaderEnd = 0;
		}
		if (mheaderEnd == 4)
		{
			misReadHeader = true;
			if (misClient)
			{
				mresponse = new Response;
				if (!mresponse->parseHeader(mstrHeader.c_str(),mstrHeader.length()))
				{
					logs->error("***** %s [CHttp::want2Read] %s parseHeader fail *****", 
						mremoteAddr.c_str(),moriUrl.c_str());
					ret = -1;
				}
			}
			else
			{
				if (mrequest == NULL)
				{
					mrequest = new Request;
					mrequest->setRemoteAddr(mremoteAddr);
				}
				if (!mrequest->parseHeader(mstrHeader.c_str(),mstrHeader.length()))
				{
					logs->error("***** %s [CHttp::want2Read] %s parseHeader fail *****", 
						mremoteAddr.c_str(),moriUrl.c_str());
					ret = -1;
				}
				moriUrl = mrequest->getUrl();
			}
		}		
	}
	if (ret != -1 && misReadHeader)
	{
		ret = msuper->doDecode();
		if (ret == CMS_ERROR)
		{
			return ret;
		}
	}
	ret = msuper->doReadData();
	msuper->down8upBytes();
	return ret;
}

int CHttp::want2Write(bool isTimeout)
{
	if (isTimeout)
	{
		assert(mcmsWriteTimeOutDo==1);
		mcmsWriteTimeOutDo--;
	}
	if (misTls)
	{
		//ssl handshake
		if (!mssl->isHandShake())
		{
			int len = 0;
			int ret = mssl->write(NULL,len);
			if (ret < 0)
			{
				return CMS_ERROR;
			}
			else if (ret == 0)
			{
				doWriteTimeout();
				return CMS_OK;
			}
		}
	}
	int ret = CMS_OK;	
	if (misClient)
	{
		if (misTls)
		{
			ret = mssl->flush();
			if (ret == CMS_ERROR)
			{
				logs->error("%s [CHttp::want2Write] %s flush fail ***",
					mremoteAddr.c_str(),moriUrl.c_str());
				return CMS_ERROR;
			}
			if (!mssl->isUsable())
			{
				//���CBufferWriter���п͹۵�����û���ͳ�ȥ��������ʱ��ʱ������ʱ�������ݣ��Ҳ��ٶ�ȡ�κ�����
				doWriteTimeout();
				return CMS_OK;
			}
		}
		else
		{
			ret = mwrBuff->flush();
			if (ret == CMS_ERROR)
			{
				logs->error("%s [CHttp::want2Write] %s flush fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),moriUrl.c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
				return CMS_ERROR;
			}
			if (!mwrBuff->isUsable())
			{
				//���CBufferWriter���п͹۵�����û���ͳ�ȥ��������ʱ��ʱ������ʱ�������ݣ��Ҳ��ٶ�ȡ�κ�����
				doWriteTimeout();
				return CMS_OK;
			}				
		}
		if (!misReadReuqest)
		{
			mstrRequestHeader = mrequest->readRequest();
			misReadReuqest = true;
		}
		if (msendRequestLen < (int)mstrRequestHeader.length())
		{
			int len = mstrRequestHeader.length() - msendRequestLen;
			if (write(mstrRequestHeader.c_str()+msendRequestLen,len) == CMS_ERROR)
			{
				logs->error("%s [CHttp::want2Write] %s send http request fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),moriUrl.c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
				return CMS_ERROR;
			}
			msendRequestLen += len;
		}						
	}
	else
	{
		if (misReadHeader)
		{
			if (misTls)
			{
				ret = mssl->flush();
				if (ret == CMS_ERROR)
				{
					logs->error("%s [CHttp::want2Write] %s flush fail ***",
						mremoteAddr.c_str(),moriUrl.c_str());
					return CMS_ERROR;
				}
				if (!mssl->isUsable())
				{
					//���CBufferWriter���пɹ۵�����û���ͳ�ȥ��������ʱ��ʱ������ʱ�������ݣ��Ҳ��ٶ�ȡ�κ�����
					doWriteTimeout();
					return CMS_OK;
				}
			}
			else
			{
				ret = mwrBuff->flush();
				if (ret == CMS_ERROR)
				{
					logs->error("%s [CHttp::want2Write] %s flush fail,errno=%d,strerrno=%s ***",
						mremoteAddr.c_str(),moriUrl.c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
					return CMS_ERROR;
				}
				if (!mwrBuff->isUsable())
				{
					//���CBufferWriter���пɹ۵�����û���ͳ�ȥ��������ʱ��ʱ������ʱ�������ݣ��Ҳ��ٶ�ȡ�κ�����
					doWriteTimeout();
					return CMS_OK;
				}				
			}
			if (mwrBuff->size() == 0 && msuper->isFinish())
			{
				reset();
				msuper->reset();
				return CMS_OK;
			}
			ret = msuper->doTransmission();
			if (ret < 0)
			{
				logs->error("***** %s [CHttp::want2Write] %s doTransmission fail *****", 
					mremoteAddr.c_str(),moriUrl.c_str());
				return CMS_ERROR;
			}
			if (ret == 0 || ret == 2)
			{
				//logs->debug("%s [CRtmpProtocol::doRtmpConnect] rtmp %s not have buffer",
				//	mremoteAddr.c_str(),getRtmpType().c_str());
				doWriteTimeout();
			}
			msuper->down8upBytes();
		}
	}
	return CMS_OK;
}

void CHttp::reset()
{
	mrequest->reset();
	mresponse->reset();
	misReadHeader = false;
	misWriteHeader = false;
	mheaderEnd = 0;
	mstrHeader.clear();

	misChunked = false;;
	mchunkedLen = 0;
	misReadChunkedLen = false;
	mchunkBytesRN.clear();
	mchunkedReadrRN = 0;

	msendRequestLen = 0;
	misReadReuqest = false;
	mstrRequestHeader.clear();
}

int CHttp::read(char **data,int &len)
{
	assert(len > 0);
	int ret = 0;
	if (misChunked)
	{
		if (!misReadChunkedLen)
		{
			//chunk size
			ret = readChunkedRN();
			if (ret <= 0)
			{
				return ret;
			}
			assert(mchunkedReadrRN == 2);
			mchunkedReadrRN = 0;
			misReadChunkedLen = true;
			mchunkedLen = hex2int64(mchunkBytesRN.c_str());
			mchunkBytesRN.clear();
		}
		if (misReadChunkedLen)
		{
			//chunk body
			char *p;
			int max;
			while (mchunkedLen > 0)
			{
				max = mchunkedLen > 1024*8 ? 1024*8 : mchunkedLen;
				if (misTls)
				{
					ret = mssl->read(&p,max);
					if (ret < 0)
					{
						return -1;
					}
					else if (ret == 0)
					{
						return 0;
					}
				}
				else
				{
					if ( mrdBuff->size() < max && mrdBuff->grow(max) == CMS_ERROR)
					{
						return -1;
					}
					if (mrdBuff->size() < max)
					{
						return 0;
					}
					p = mrdBuff->readBytes(max);
				}
				mbyteReadWrite->writeBytes(p,max);
				mchunkedLen -= max;
			}
			if (mchunkedLen == 0)
			{
				//chunked footer;
				ret = readChunkedRN();
				if (ret <= 0)
				{
					return ret;
				}
				assert(mchunkedReadrRN == 2);
				mchunkedReadrRN = 0;
				misReadChunkedLen = false;
				mchunkBytesRN.clear();
			}
		}
		if (mbyteReadWrite->size() >= len)
		{
			*data = mbyteReadWrite->readBytes(len);
			return len;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (misTls)
		{
			ret = mssl->read(data,len);
			if (ret < 0)
			{
				return -1;
			}
			else if (ret == 0)
			{
				return 0;
			}
		}
		else
		{
			if ( mrdBuff->size() < len && mrdBuff->grow(len) == CMS_ERROR)
			{
				return -1;
			}
			if (mrdBuff->size() < len)
			{
				return 0;
			}
			*data = mrdBuff->readBytes(len);
		}
	}
	return len;
}

int CHttp::readChunkedRN()
{
	int ret = 0;
	do 
	{
		char *p = NULL;
		char ch;
		int  max = 1;
		if (misTls)
		{
			ret = mssl->read(&p,max);
			if (ret < 0)
			{
				ret = -1;
				break;
			}
			else if (ret == 0)
			{
				ret = 0;
				break;
			}
			ch = *p;
		}
		else
		{
			if ( mrdBuff->size() < 1 && mrdBuff->grow(1) == CMS_ERROR)
			{
				logs->error("%s [CHttp::readChunkedRN] %s http header read one byte fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),moriUrl.c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
				ret = -1;
				break;
			}
			if (mrdBuff->size() < 1)
			{
				ret = 0;
				break;
			}			
			ch = mrdBuff->readByte();
		}
		mchunkBytesRN.append(1,ch);
		if (ch == '\r')
		{
			if (mchunkedReadrRN % 2 == 0)
			{
				mchunkedReadrRN++;
			}
			else
			{
				logs->warn("##### %s [CHttp::read] %s 1 read header unexpect #####", 
					mremoteAddr.c_str(),moriUrl.c_str());
				mchunkedReadrRN = 0;
			}
		}
		else if (ch == '\n')
		{
			if (mchunkedReadrRN % 2 == 1)
			{
				mchunkedReadrRN++;
			}
			else
			{
				logs->warn("##### %s [CHttp::read] %s 2 read header unexpect #####", 
					mremoteAddr.c_str(),moriUrl.c_str());
				mchunkedReadrRN = 0;
			}
		}
		else
		{
			mchunkedReadrRN = 0;
		}
		if (mchunkedReadrRN == 2)
		{
			break;
		}
	} while (true);
	return ret;
}

int CHttp::write(const char *data,int &len)
{
	assert(len > 0);
	int leftSize = 0;
	if (misTls)
	{
		int ret = mssl->write(data,len);
		if (ret < 0)
		{
			return CMS_ERROR;
		}
		else if (ret == 0)
		{
			return CMS_OK;
		}
		leftSize = mssl->bufferWriteSize();
	}
	else
	{
		int ret = mwrBuff->writeBytes(data,len);
		if (ret == CMS_ERROR)
		{
			return CMS_ERROR;
		}
		leftSize = mwrBuff->size();
	}
	if (leftSize > 0)
	{
		//�첽
		//ev_io_init(msuper->evWriteIO(), writeEV, mrw->fd(), EV_WRITE);		
		//ev_io_start(msuper->evLoop(), msuper->evWriteIO());
		msuper->evWriteIO();
	}
	else if (!misClient)
	{
		//ev_io_stop(msuper->evLoop(), msuper->evWriteIO());
		doWriteTimeout();
	}
	return CMS_OK;
}

void CHttp::syncIO()
{
	int leftSize = 0;
	if (misTls)
	{
		leftSize = mssl->bufferWriteSize();
	}
	else
	{
		leftSize = mwrBuff->size();
	}
	if (leftSize > 0)
	{
		//�첽
		//ev_io_init(msuper->evWriteIO(), writeEV, mrw->fd(), EV_WRITE);		
		//ev_io_start(msuper->evLoop(), msuper->evWriteIO());
		msuper->evWriteIO();
	}
	else if (!misClient)
	{
		//ev_io_stop(msuper->evLoop(), msuper->evWriteIO());
		msuper->evWriteIO();
		doWriteTimeout();
	}
}

void CHttp::setChunked()
{
	misChunked = true;
	mbyteReadWrite = new CByteReaderWriter();
}

int CHttp::sendMetaData(Slice *s)
{
	*mbinaryWriter << (char)FLV_TAG_SCRIPT;
	*mbinaryWriter << (char)(s->miDataLen >> 16);
	*mbinaryWriter << (char)(s->miDataLen >> 8);
	*mbinaryWriter << (char)(s->miDataLen);
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	if (msuper->sendBefore(mbinaryWriter->getData(),mbinaryWriter->getLength()) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendMetaData] %s 1 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	mbinaryWriter->reset();
	if (msuper->sendBefore(s->mData,s->miDataLen) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendMetaData] %s 2 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	*mbinaryWriter << (char)((s->miDataLen+11) >> 24);
	*mbinaryWriter << (char)((s->miDataLen+11) >> 16);
	*mbinaryWriter << (char)((s->miDataLen+11) >> 8);
	*mbinaryWriter << (char)((s->miDataLen+11));
	if (msuper->sendBefore(mbinaryWriter->getData(),mbinaryWriter->getLength()) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendMetaData] %s 3 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	mbinaryWriter->reset();
	return CMS_OK;

}

int CHttp::sendVideoOrAudio(Slice *s,uint32 uiTimestamp)
{
	if (s->miDataType == DATA_TYPE_AUDIO || s->miDataType == DATA_TYPE_FIRST_AUDIO)
	{
		*mbinaryWriter << (char)FLV_TAG_AUDIO;
	}
	else if (s->miDataType == DATA_TYPE_VIDEO || s->miDataType == DATA_TYPE_FIRST_VIDEO)
	{
		*mbinaryWriter << (char)FLV_TAG_VIDEO;
	}
	else
	{
		assert(0);
	}
	*mbinaryWriter << (char)(s->miDataLen >> 16);
	*mbinaryWriter << (char)(s->miDataLen >> 8);
	*mbinaryWriter << (char)(s->miDataLen);
	*mbinaryWriter << (char)(uiTimestamp >> 16);
	*mbinaryWriter << (char)(uiTimestamp >> 8);
	*mbinaryWriter << (char)(uiTimestamp);
	*mbinaryWriter << (char)(uiTimestamp >> 24);
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	*mbinaryWriter << (char)0x00;
	if (msuper->sendBefore(mbinaryWriter->getData(),mbinaryWriter->getLength()) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendVideoOrAudio] %s 1 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	mbinaryWriter->reset();
	if (msuper->sendBefore(s->mData,s->miDataLen) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendVideoOrAudio] %s 2 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	*mbinaryWriter << (char)((s->miDataLen+11) >> 24);
	*mbinaryWriter << (char)((s->miDataLen+11) >> 16);
	*mbinaryWriter << (char)((s->miDataLen+11) >> 8);
	*mbinaryWriter << (char)((s->miDataLen+11));
	if (msuper->sendBefore(mbinaryWriter->getData(),mbinaryWriter->getLength()) == CMS_ERROR)
	{
		logs->error("%s [CHttp::sendVideoOrAudio] %s 3 sendBefore fail ***",
			mremoteAddr.c_str(),moriUrl.c_str());
		return CMS_ERROR;
	}
	mbinaryWriter->reset();
	return CMS_OK;
}

int CHttp::writeBuffSize()
{
	int leftSize = 0;
	if (misTls)
	{
		leftSize = mssl->bufferWriteSize();
	}
	else
	{
		leftSize = mwrBuff->size();
	}
	return leftSize;
}

void CHttp::setWriteBuffer(int size)
{
	mrw->setWriteBuffer(size);
}

std::string CHttp::remoteAddr()
{
	return mremoteAddr;
}

std::string CHttp::getUrl()
{
	return mrequest->getUrl();
}

bool CHttp::setUrl(std::string url)
{	
	mrequest = new Request;
	if (!mrequest->setUrl(url))
	{
		return false;
	}
	mrequest->setRemoteAddr(mremoteAddr);
	mrequest->setRefer(mreferUrl);
	return true;
}

void CHttp::setRefer(std::string refer)
{
	mreferUrl = refer;
}

void CHttp::setOriUrl(std::string oriUrl)
{
	if (moriUrl.empty())
	{
		moriUrl = oriUrl;
	}
}

void CHttp::doWriteTimeout()
{
	if (mcmsWriteTimeout == NULL)
	{
		mcmsWriteTimeout = mallcoCmsTimer();
		cms_timer_init(mcmsWriteTimeout,mrw->fd(),wait2WriteEV);
		assert(mcmsWriteTimeOutDo == 0);
		mcmsWriteTimeOutDo++;
		cms_timer_start(mcmsWriteTimeout);
	}
	else
	{
		if (mcmsWriteTimeOutDo == 0)
		{
			mcmsWriteTimeOutDo++;
			cms_timer_start(mcmsWriteTimeout);			
		}		
	}
}

void CHttp::doReadTimeout()
{
	if (mcmsReadTimeout == NULL)
	{
		mcmsReadTimeout = mallcoCmsTimer();
		cms_timer_init(mcmsReadTimeout,mrw->fd(),wait2ReadEV,false);
		assert(mcmsReadTimeOutDo == 0);
		mcmsReadTimeOutDo++;
		cms_timer_start(mcmsReadTimeout,false);
	}
	else
	{
		if (mcmsReadTimeOutDo == 0)
		{
			mcmsReadTimeOutDo++;
			cms_timer_start(mcmsReadTimeout,false);			
		}		
	}
}

Request *CHttp::httpRequest()
{
	return mrequest;
}

Response *CHttp::httpResponse()
{
	if (mresponse == NULL)
	{
		mresponse = new Response;
	}
	return mresponse;
}

cms_timer *CHttp::cmsTimer2Write()
{
	return mcmsWriteTimeout;
}

cms_timer *CHttp::cmsTimer2Read()
{
	return mcmsReadTimeout;
}
