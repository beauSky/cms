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
#include <net/cms_tcp_conn.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <dnscache/cms_dns_cache.h>
#include <core/cms_errno.h>
#include <s2n/s2n.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <netinet/tcp.h>


TCPConn::TCPConn(int fd)
{
	mreadTimeout = 1;
	mreadBytes = 0;
	mwriteTimetou = 1;
	mwriteBytes = 0;
	mfd = fd;
	struct sockaddr_in from;
	socklen_t len = sizeof(from);
	if(getpeername(fd, (struct sockaddr *)&from, &len) != -1)
	{
		char szIP[16] = {0};
		ipInt2ipStr(from.sin_addr.s_addr,szIP);
		char tmp[23] = {0};
		snprintf(tmp,sizeof(tmp),"%s:%d",szIP,ntohs(from.sin_port));
		maddr = tmp;
	}
	else
	{
		logs->error("*** [TCPConn::TCPConn] getpeername fail,errno=%d,strerr=%s ***",errno,strerror(errno));
	}
}

TCPConn::TCPConn()
{
	mreadTimeout = 1;
	mreadBytes = 0;
	mwriteTimetou = 1;
	mwriteBytes = 0;
	mfd = -1;
	merrcode = 0;
}

TCPConn::~TCPConn()
{
	if (mfd != -1)
	{
		::close(mfd);
		mfd = -1;
	}
}


int   TCPConn::dialTcp(char *addr,ConnType connectType)
{
	mconnectType = connectType;	

	unsigned short port;
	char *host,*pos;	
	pos = (char*)strchr(addr,':');
	if (pos == NULL)
	{
		logs->error("*** TCPConn dialTcp addr %s is illegal *****",addr);
		return CMS_ERROR;
	}
	host = (char*)addr;
	*pos++ = '\0';
	port = (unsigned short)atoi(pos);
	mfd = socket(AF_INET,SOCK_STREAM,0);
	if (mfd == CMS_INVALID_SOCK)
	{
		logs->error("*** TCPConn dialTcp create socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
		return CMS_ERROR;
	}
	memset(&mto, 0, sizeof(mto));
	mto.sin_family = AF_INET;
	mto.sin_port = htons(port);
	unsigned long ip;
	if (!CDnsCache::instance()->host2ip(host,ip))
	{
		logs->error("*** TCPConn dialTcp dns cache error *****");
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	mto.sin_addr.s_addr = ip;
	char szAddr[25] = {0};
	ipInt2ipStr(ip,szAddr);
	snprintf(szAddr+strlen(szAddr),sizeof(szAddr)-strlen(szAddr),":%d",port);
	maddr = szAddr;
	logs->info("##### TCPConn dialTcp addr %s fd=%d #####",maddr.c_str(),mfd);
	return CMS_OK;
}

int	  TCPConn::connect()
{
	nonblocking(mfd);
	if (::connect(mfd,(struct sockaddr *)&mto, sizeof(mto)) < 0)
	{
		if (errno != EINPROGRESS)
		{
			logs->error("*** [TCPConn::connect] connect socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
			::close(mfd);
			return CMS_ERROR;
		}		
	}
	logs->info("##### TCPConn connect addr %s succ #####",maddr.c_str());
	return CMS_OK;
}

int   TCPConn::read(char* dstBuf,int len,int &nread)
{
	int nbRead = recv(mfd,dstBuf,len,0);
	if (nbRead <= 0)
	{		
		if (nbRead < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
			return CMS_OK;
		}
		if (nbRead == 0)
		{
			merrcode = CMS_ERRNO_FIN;
		}
		else
		{
			merrcode = errno;
		}
		return CMS_ERROR;
	}
	mreadBytes += (long long)nbRead;
	nread = nbRead;
	return CMS_OK;
}

int   TCPConn::write(char *srcBuf,int len,int &nwrite)
{
	int nbWrite = send(mfd,srcBuf,len,0);	
	if (nbWrite <= 0)
	{
		if (nbWrite < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
			return CMS_OK;
		}
		if (nbWrite == 0)
		{
			merrcode = CMS_ERRNO_FIN;
		}
		else
		{
			//logs->info("##### TCPConn write addr %s fail,fd=%d #####",maddr.c_str(),mfd);
			merrcode = errno;
		}
		return CMS_ERROR;
	}
	mwriteBytes += (long long)nbWrite;
	nwrite = nbWrite;
	return CMS_OK;
}

char  *TCPConn::errnoCode()
{
	return cmsStrErrno(merrcode);
}

int  TCPConn::errnos()
{
	return merrcode;
}

int TCPConn::setNodelay(int on)
{
	return setsockopt(mfd, IPPROTO_TCP, TCP_NODELAY, (void *)&on, sizeof(on));
}

int	TCPConn::setReadBuffer(int size)
{
	return setsockopt(mfd, IPPROTO_TCP, SO_RCVBUF, (void *)&size, sizeof(size));
}

int	TCPConn::setWriteBuffer(int size)
{
	return setsockopt(mfd, IPPROTO_TCP, SO_SNDBUF, (void *)&size, sizeof(size));
}

int   TCPConn::remoteAddr(char *addr,int len)
{
	memcpy(addr,maddr.c_str(),cmsMin(len,(int)maddr.length())); 
	return CMS_OK;
}

void  TCPConn::setReadTimeout(long long readTimeout)
{
	mreadTimeout = readTimeout;
}

long long TCPConn::getReadTimeout()
{
	return mreadTimeout;
}

void  TCPConn::setWriteTimeout(long long writeTimeout)
{
	mwriteTimetou = writeTimeout;
}

long long TCPConn::getWriteTimeout()
{
	return mwriteTimetou;
}

long long TCPConn::getReadBytes()
{
	return mreadBytes;
}

long long TCPConn::getWriteBytes()
{
	return mwriteBytes;
}

void TCPConn::close()
{
	if (mfd != -1)
	{
		::close(mfd);
		mfd = -1;
	}
}

ConnType TCPConn::connectType()
{
	return mconnectType;
}

int TCPConn::fd()
{
	return mfd;
}


TCPListener::TCPListener()
{
	mruning = false;
	mfd = -1;
}

int  TCPListener::listen(char* addr,ConnType listenType)
{
	mlistenAddr = addr;
	struct sockaddr_in serv_addr;
	int n;
	unsigned short port;
	char *host,*pos;
	logs->info("##### TCPListener listen addr %s #####",addr);
	pos = (char*)strchr(addr,':');
	if (pos == NULL)
	{
		logs->error("*** TCPListener listen addr %s is illegal *****",addr);
		return CMS_ERROR;
	}
	host = (char*)addr;
	*pos++ = '\0';
	if (strlen(host) == 0)
	{
		host = (char *)"0.0.0.0";
	}
	port = (unsigned short)atoi(pos);
	mfd = socket(AF_INET,SOCK_STREAM,0);
	if (mfd == CMS_INVALID_SOCK)
	{
		logs->error("*** TCPListener listen create socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
		return CMS_ERROR;
	}
	n = 1;
	if (setsockopt(mfd, SOL_SOCKET, SO_REUSEADDR, (char *)&n, sizeof(n)) < 0)
	{
		logs->error("*** TCPListener listen set SO_REUSEADDR fail,errno=%d,errstr=%s *****",errno,strerror(errno));
	}
	nonblocking(mfd);
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	unsigned long ip;
	if (!CDnsCache::instance()->host2ip(host,ip))
	{
		logs->error("*** TCPListener listen dns cache error *****");
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	serv_addr.sin_addr.s_addr = ip;
	if (bind(mfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		logs->error("*** TCPListener listen bind %s socket is error,errno=%d,errstr=%s *****",mlistenAddr.c_str(),errno,strerror(errno));
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	if (::listen(mfd, 256) < 0)
	{	
		logs->error("*** TCPListener listen ::listen socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	mruning = true;
	//createThread(this);
	mlistenType = listenType;
	if (mlistenType == TypeHttps)
	{
		if (s2n_init() < 0) 
		{
			logs->error("*** TCPListener error running s2n_init(): '%s' ***", s2n_strerror(s2n_errno, "EN"));
			::close(mfd);
			mfd = -1;
			return CMS_ERROR;
		}
		//还需要设置证书
	}
	return CMS_OK;
}

ConnType TCPListener::listenType()
{
	return mlistenType;
}

void TCPListener::stop()
{
	logs->info("### TCPListener begin stop listening %s ###",mlistenAddr.c_str());
	if (mruning)
	{
		mruning = false;
		::close(mfd);
		mfd = -1;
	}
	logs->info("### TCPListener finish stop listening %s ###",mlistenAddr.c_str());
}

int  TCPListener::accept()
{
	int cnfd = -1;
	struct sockaddr_in from;
	socklen_t fromlen;
	fromlen = sizeof(from);
	cnfd = ::accept(mfd, (struct sockaddr *)&from, &fromlen);
	if (cnfd == -1) {
		logs->error("*** TCPListener can't accept connection err=%d,strerr=%s ***",errno,strerror(errno));
		return CMS_ERROR;
	}
	return cnfd;
}

int  TCPListener::fd()
{
	return mfd;
}


