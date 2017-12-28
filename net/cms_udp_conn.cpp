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
#include <net/cms_udp_conn.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <config/cms_config.h>
#include <dnscache/cms_dns_cache.h>
#include <ev/cms_ev.h>
#include <core/cms_errno.h>
#include <s2n/s2n.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <time.h>
#include <queue>
#include <set>
#include <map>


#define MapUdpConnIter map<UdpAddr,UdpConnInfo*>::iterator	
#define ListUdpConnIter list<UDPConn *>::iterator

#define UDP_BASE_PORT	20000 // minimum port for listening
#define UDP_MAX_PORT	65535 // maximum port for listening

long long gbasePortInt = getNsTime();
CLock	  glockBasePortInt;

queue<int> gudpSocket;  //用于分配给udp listener接收的新的连接，socket为负数，用于和系统的区分开
set<int>   gudpOnly;
CLock	   glockUdpSocket;

unsigned int gconvID = 0;
CLock mlockConvID;

#define MapSockUdpConnIter map<int, UDPConn *>::iterator
map<int, UDPConn *> gudpConnMap[TIME_UDP_THREAD_NUM];
CRWlock gudpConnRWLock[TIME_UDP_THREAD_NUM];


int output(const char *buf, int len, struct IKCPCB *kcp, void *user)
{
	char szIP[25] = {0};
	ipInt2ipStr(kcp->maddr.sin_addr.s_addr,szIP);
// 	int port = ntohs(kcp->maddr.sin_port);
// 	logs->debug("write sock %d addr %s:%d.len=%d",kcp->fd,szIP,port,len);
	return sendto(kcp->fd,buf,len,0,(const sockaddr *)&kcp->maddr,sizeof(kcp->maddr));
}

unsigned int getConvID()
{
	if (gconvID == 0)
	{
		srand(time(NULL));
		mlockConvID.Lock();
		gconvID = (unsigned int)rand();
		mlockConvID.Unlock();
	}
	unsigned int conv;
	mlockConvID.Lock();
	conv = gconvID++;
	mlockConvID.Unlock();
	return conv;
}

void initUdpSocket()
{
	glockUdpSocket.Lock();
	if (gudpSocket.empty())
	{
		for (int i  = -2; i > CConfig::instance()->udpFlag()->udpConnNum()*-1; i--)
		{
			gudpSocket.push(i);
			gudpOnly.insert(i);
		}
	}
	glockUdpSocket.Unlock();
}

void pushUdpSock(int sock)
{
	if (sock >= 0)
	{
		return;
	}
	glockUdpSocket.Lock();
	set<int>::iterator it = gudpOnly.find(sock);
	if (it != gudpOnly.end())
	{
		logs->warn("#####@@@@@ [never]gudpOnly push same sock %d @@@@@#####",sock);
	}
	else
	{
		gudpOnly.insert(sock);
		gudpSocket.push(sock);
	}
	glockUdpSocket.Unlock();
	return;
}

int popUdpSock()
{
	int sock = 0;
	glockUdpSocket.Lock();
	if (gudpSocket.empty())
	{
		logs->warn("#####@@@@@ [never]gudpOnly pop all the sock has been pop @@@@@#####");
	}
	else
	{
		sock = gudpSocket.front();
		gudpSocket.pop();
		set<int>::iterator it = gudpOnly.find(sock);
		if (it != gudpOnly.end())
		{
			gudpOnly.erase(it);
		}
		else
		{
			logs->warn("#####@@@@@ [never]gudpOnly pop not find the pop sock %d @@@@@#####",sock);
			cmsSleep(1000);
			assert(0);
		}
	}
	glockUdpSocket.Unlock();
	return sock;
}

void pushUDPConn(UDPConn *conn)
{
	int i = conn->fd();
	if (i < 0)
	{
		i = -i;
	}
	i = i % TIME_UDP_THREAD_NUM;
	gudpConnRWLock[i].WLock();
	MapSockUdpConnIter it = gudpConnMap[i].find(conn->fd());
	assert(it == gudpConnMap[i].end());
	gudpConnMap[i].insert(make_pair(conn->fd(), conn));
	gudpConnRWLock[i].UnWLock();
}

void popUDPConn(UDPConn *conn)
{
	int i = conn->fd();
	if (i < 0)
	{
		i = -i;
	}
	i = i % TIME_UDP_THREAD_NUM;
	gudpConnRWLock[i].WLock();
	MapSockUdpConnIter it = gudpConnMap[i].find(conn->fd());
	if (it != gudpConnMap[i].end())
	{
		gudpConnMap[i].erase(it);
	}	
	gudpConnRWLock[i].UnWLock();
}

void udpTickerCallBack(void *t)
{
	cms_udp_timer *ct = (cms_udp_timer *)t;
	int i = ct->fd;
	if (i < 0)
	{
		i = -i;
	}
	i = i % TIME_UDP_THREAD_NUM;
	gudpConnRWLock[i].RLock();
	MapSockUdpConnIter it = gudpConnMap[i].find(ct->fd);
	if (it != gudpConnMap[i].end())
	{
		if (it->second->flushW(ct->uid) == CMS_ERROR)
		{
			//uid对不上 旧的计时器
			atomicUdpDec(ct);
		}
	}
	else
	{
		//找不到可能被删除了
		atomicUdpDec(ct);
	}
	gudpConnRWLock[i].UnRLock();
}

UDPConn::UDPConn(UdpAddr ua,IUINT32 conv,int fd,unsigned long ipInt,unsigned short port,bool isListen,ConnType connectType)
{
	mwatcherWriteIO = NULL;
	mreadTimeout = 1;
	mreadBytes = 0;
	mwriteTimetou = 1;
	mwriteBytes = 0;	
	merrcode = 0;
	mfd = 0;
	mlsFd = -1;
	misListen = isListen;
	if (isListen)
	{
		mlsFd = fd;
		mfd = popUdpSock();
	}
	else
	{
		mfd = fd;
	}
	mkcp = ikcp_create(conv,fd,ipInt,port,NULL);
	ikcp_setoutput(mkcp,output);
	ikcp_setoutput(mkcp,output);
	ikcp_setmtu(mkcp,1400);
	ikcp_wndsize(mkcp,128,128);
	ikcp_nodelay(mkcp,1,50,0,1);

	mconnectType = connectType;

	char szAddr[25] = {0};
	ipInt2ipStr(ipInt,szAddr);
	snprintf(szAddr+strlen(szAddr),sizeof(szAddr)-strlen(szAddr),":%d",port);
	mraddr = szAddr;
	mua = ua;
	logs->info("##### UDPConn new UDP addr %s fd=%d #####",mraddr.c_str(),mfd);
	mwriteTotalLen = 0;
	mtimer = NULL;
	muid = 0;
	mtickerDo = 0;
	ticker();
}

UDPConn::UDPConn()
{
	mwatcherWriteIO = NULL;
	mreadTimeout = 1;
	mreadBytes = 0;
	mwriteTimetou = 1;
	mwriteBytes = 0;
	mfd = 0;
	mlsFd = -1;
	merrcode = 0;
	misListen = false;
	mkcp = NULL;
	mwriteTotalLen = 0;
	mtimer = NULL;
	muid = 0;
	mtickerDo = 0;
}

UDPConn::~UDPConn()
{
	if (mfd > 0)
	{
		popUDPConn(this);
		::close(mfd);
		mfd = 0;
	}
	else if (mfd < 0)
	{
		popUDPConn(this);
		pushUdpSock(mfd);
	}
	if (mkcp)
	{
		ikcp_release(mkcp);
	}
	if (mwatcherWriteIO)
	{
		freeCmsNetEv(mwatcherWriteIO);
	}
	if (mtimer)
	{
		freeCmsUdpTimer(mtimer);
	}
	UdpMsg *um;
	mlockUdpMsg.Lock();
	for (;!mqueueUdpMsg.empty();)
	{
		um = mqueueUdpMsg.front();
		mqueueUdpMsg.pop();
		if (um->mdata)
		{
			delete[] um->mdata;
		}
		delete um;
	}
	mlockUdpMsg.Unlock();
}


int UDPConn::dialUdp(char *addr,ConnType connectType)
{
	std::string sAddr;
	sAddr.append(addr);
	mconnectType = connectType;	
	std::string strHost;
	unsigned short port;
	size_t pos = sAddr.find(":");
	if (pos == std::string::npos)
	{
		logs->error("*** UDPConn dialUdp addr %s is illegal *****",addr);
		return CMS_ERROR;
	}
	strHost = sAddr.substr(0,pos);
	port = (unsigned short)atoi(sAddr.substr(pos+1).c_str());
	mfd = socket(AF_INET,SOCK_DGRAM,0);
	if (mfd == CMS_INVALID_SOCK)
	{
		logs->error("*** UDPConn dialTcp create socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
		return CMS_ERROR;
	}
	ticker();
	nonblocking(mfd);
	int bindTimes = 0;
	struct sockaddr_in to;
	memset(&to, 0, sizeof(to));
	int bindPort = 0;
	do
	{
		glockBasePortInt.Lock();
		bindPort = UDP_BASE_PORT + (int)(gbasePortInt % (UDP_MAX_PORT-UDP_BASE_PORT));
		gbasePortInt++;
		if (gbasePortInt == 0x7FFFFFFF)
		{
			gbasePortInt = 0;
		}
		glockBasePortInt.Unlock();

		to.sin_family = AF_INET;
		to.sin_port = htons(bindPort);
		to.sin_addr.s_addr = 0;
		if (bind(mfd, (struct sockaddr *)&to, sizeof(to)) < 0)
		{
			bindTimes++;
			continue;
		}
		break;
	}while(bindTimes < 10000);
	if (bindTimes >= 10000)
	{
		logs->error("*** UDPListener dialUdp bind port error,errno=%d,errstr=%s *****",errno,strerror(errno));
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}

	memset(&mto, 0, sizeof(mto));
	mto.sin_family = AF_INET;
	mto.sin_port = htons(port);
	unsigned long ip;
	if (!CDnsCache::instance()->host2ip(strHost.c_str(),ip))
	{
		logs->error("*** UDPConn dialTcp dns cache error *****");
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	mto.sin_addr.s_addr = ip;
	
	mraddr = addr;
	char szAddr[25] = {0};
	snprintf(szAddr,sizeof(szAddr),":%d",bindPort);
	mladdr = szAddr;
	logs->info("##### UDPConn dialTcp %s:%d fd=%d #####",strHost.c_str(),port,mfd);

	mkcp = ikcp_create(getConvID(),mfd,ip,port,NULL);
	ikcp_setoutput(mkcp,output);
	ikcp_setmtu(mkcp,1400);
	ikcp_wndsize(mkcp,128,128);
	ikcp_nodelay(mkcp,1,50,0,1);
	

	mua.miBindPort = bindPort;
	mua.miPort = port;
	mua.mlistener = NULL;
	mua.msAddr = mraddr;
	mua.muiAddr = ip;

	return CMS_OK;
}

int	  UDPConn::connect()
{	
	if (::connect(mfd,(struct sockaddr *)&mto, sizeof(mto)) < 0)
	{
		if (errno != EINPROGRESS)
		{
			logs->error("*** [UDPConn::connect] connect socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
			::close(mfd);
			return CMS_ERROR;
		}
	}
	logs->info("##### UDPConn connect addr %s succ #####",mraddr.c_str());
	return CMS_OK;
}

int UDPConn::read(char* dstBuf,int len,int &nread)
{
	int ret = CMS_OK;
	nread = 0;
	mlockKcp.Lock();
	//
	//判断是否已经断开 暂时没加
	//
	nread = ikcp_recv(mkcp,dstBuf,len);
	if (nread < 0) 
	{
		logs->error("***** UDPConn read addr %s fd=%d ikcp_recv handle fail *****",mraddr.c_str(),mfd); 
		merrcode = CMS_KCP_ERR_FAIL;
		ret = CMS_ERROR;
		nread = 0;
	}
	mlockKcp.Unlock();
	return ret;
}

int   UDPConn::write(char *srcBuf,int len,int &nwrite)
{
	int ret = CMS_OK;
	if (len <= 0)
	{
		return ret;
	}	
	nwrite = 0;
	int nbLeft = len;
	mlockKcp.Lock();
	for (;nwrite < len;)
	{
		//
		//判断是否已经断开 暂时没加
		//
		if (ikcp_waitsnd(mkcp) < (int)mkcp->snd_wnd)
		{
			int max = (int)((mkcp->mss<<8)-mkcp->mss);
			if (nbLeft <= max)
			{
				if (ikcp_send(mkcp,srcBuf+nwrite,nbLeft) == 0)
				{	
					mwriteTotalLen += nbLeft;
					nwrite += nbLeft;
					nbLeft = 0;					
					break;
				}
				else
				{
					logs->error("***** UDPConn write addr %s fd=%d ikcp_send handle fail *****",mraddr.c_str(),mfd); 
					merrcode = CMS_KCP_ERR_FAIL;
					ret = CMS_ERROR;
					break;
				}
			}
			else
			{
				if (ikcp_send(mkcp,srcBuf+nwrite,max) == 0)
				{			
					mwriteTotalLen += max;
					nwrite += max;
					nbLeft -= max;					
				}
				else
				{
					logs->error("***** 2 UDPConn write addr %s fd=%d ikcp_send handle fail *****",mraddr.c_str(),mfd); 
					merrcode = CMS_KCP_ERR_FAIL;
					ret = CMS_ERROR;
					break;
				}
				//判断中途是否被close 暂时没加
			}
		}
		else
		{
// 			logs->error("***** kcp snd window size not enough *****");
			break;
		}
	}
	if (nwrite > 0)
	{
		ikcp_flush(mkcp);
	}
// 	logs->error("udp send total len=%d", mwriteTotalLen);
	mlockKcp.Unlock();
	return ret;
}

char  *UDPConn::errnoCode()
{
	return cmsStrErrno(merrcode);
}

int  UDPConn::errnos()
{
	return merrcode;
}

int UDPConn::setNodelay(int on)
{
	if (mfd > 0)
	{
		return setsockopt(mfd, IPPROTO_TCP, TCP_NODELAY, (void *)&on, sizeof(on));
	}
	return 0;
}

int	UDPConn::setReadBuffer(int size)
{
	if (mfd > 0)
	{
		return setsockopt(mfd, IPPROTO_TCP, SO_RCVBUF, (void *)&size, sizeof(size));
	}
	return 0;
}

int	UDPConn::setWriteBuffer(int size)
{
	if (mfd > 0)
	{
		return setsockopt(mfd, IPPROTO_TCP, SO_SNDBUF, (void *)&size, sizeof(size));
	}
	return 0;
}

int UDPConn::remoteAddr(char *addr,int len)
{
	memcpy(addr,mraddr.c_str(),cmsMin(len,(int)mraddr.length())); 
	return CMS_OK;
}

int UDPConn::localAddr(char *addr,int len)
{
	memcpy(addr,mladdr.c_str(),cmsMin(len,(int)mladdr.length())); 
	return CMS_OK;
}

void  UDPConn::setReadTimeout(long long readTimeout)
{
	mreadTimeout = readTimeout;
}

long long UDPConn::getReadTimeout()
{
	return mreadTimeout;
}

void  UDPConn::setWriteTimeout(long long writeTimeout)
{
	mwriteTimetou = writeTimeout;
}

long long UDPConn::getWriteTimeout()
{
	return mwriteTimetou;
}

long long UDPConn::getReadBytes()
{
	return mreadBytes;
}

long long UDPConn::getWriteBytes()
{
	return mwriteBytes;
}

void UDPConn::close()
{
	popUDPConn(this);
	if (mfd > 0)
	{
		::close(mfd);
		mfd = 0;
	}
	else if (mfd < 0)
	{
		pushUdpSock(mfd);
		mfd = 0;
	}	
}

ConnType UDPConn::connectType()
{
	return mconnectType;
}

int UDPConn::fd()
{
	return mfd;
}

void UDPConn::recvData()
{
	
}

int UDPConn::flushR()
{
	if (mfd > 0)
	{
		struct sockaddr_in addr;
		socklen_t addrLen = sizeof(addr);
		int ret;
		char *data = NULL;
		IUINT32 conv;
		int len = MTU_LIMIT;
		data = new char[MTU_LIMIT];

		bool isFullWndB = false;
		bool isNFullWndE = false;

		mlockKcp.Lock();
		if (ikcp_waitsnd(mkcp) >= (int)mkcp->snd_wnd)
		{
			isFullWndB = true;
		}
		mlockKcp.Unlock();

		do
		{	
			ret = recvfrom(mfd, data, len, 0, (struct sockaddr*)&addr, &addrLen);
			if (ret >= (int)IKCP_OVERHEAD)
			{
				int r = ikcp_check_legal(data,ret,conv);
				if (r == 0)
				{
					//数据合法 允许处理
					mlockKcp.Lock();
					ret = ikcp_input(mkcp,data,ret);			
					mlockKcp.Unlock();					

					if (ret != 0)
					{
						logs->error("***** UDPConn flushR addr %s fd=%d ikcp_input handle fail %d *****",mraddr.c_str(),mfd, ret); 
						merrcode = CMS_KCP_ERR_FAIL;
						delete[] data;
						return CMS_ERROR;
					}
				}
				else
				{
					logs->debug(" UDPConn flushR addr %s fd=%d udp header is not enought ",mraddr.c_str(),mfd); 
				}
			}
			else if (ret < (int)IKCP_OVERHEAD && ret > 0)
			{
				logs->debug(" UDPConn flushR addr %s fd=%d ienter IKCP_OVERHEAD error len=%d",mraddr.c_str(),mfd,ret); 
			}
			else if (ret == -1)
			{
				delete[] data;
				data = NULL;
				break;
			}
		}while (true);

		mlockKcp.Lock();
		if (ikcp_waitsnd(mkcp) < (int)mkcp->snd_wnd)
		{
			isNFullWndE = true;
		}
		mlockKcp.Unlock();

		if (isFullWndB && isNFullWndE && mwatcherWriteIO != NULL)
		{
			//缓冲区变成可写
			mwatcherWriteIO->mcallBack(mwatcherWriteIO, EventWrite);
			printf(">>>>>>11111 UDPConn::flushW 1 fd=%d write event\n", mfd);
		}
	}
	else
	{
		int ret;
		UdpMsg *um = NULL;		
		while (popUM(&um))
		{
			mlockKcp.Lock();
			ret = ikcp_input(mkcp,um->mdata,um->mlen);			
			mlockKcp.Unlock();

			delete []um->mdata;
			delete um;

			if (ret != 0)
			{
				logs->error("***** UDPConn flushR addr %s fd=%d ikcp_input handle fail %d *****",mraddr.c_str(),mfd, ret); 
				merrcode = CMS_KCP_ERR_FAIL;
				return CMS_ERROR;
			}
		}		
	}
	return CMS_OK;
}

int UDPConn::flushW(uint64 uid)
{
	if (uid != muid)
	{
		return CMS_ERROR;
	}
	printf(">>>>>>11111 UDPConn::flushW 1 fd=%d enter,mwatcherWriteIO is null %s,waitsnd=%d,snd=%d\n", 
		mfd, mwatcherWriteIO?"false":"true", ikcp_waitsnd(mkcp), mkcp->snd_wnd);
// 	bool isFullWndB = false;
// 	bool isNFullWndE = false;
	mlockKcp.Lock();	

// 	if (ikcp_waitsnd(mkcp) >= (int)mkcp->snd_wnd)
// 	{
// 		isFullWndB = true;		
// 	}

	ikcp_update(mkcp,getTickCount());
	//ikcp_flush(mkcp);

// 	if (ikcp_waitsnd(mkcp) < (int)mkcp->snd_wnd)
// 	{
// 		isNFullWndE = true;
// 	}
	mlockKcp.Unlock();

// 	if (isFullWndB && isNFullWndE)
// 	{
// 		//缓冲区变成可写
// 		mwatcherWriteIO->mcallBack(mwatcherWriteIO,EventWrite);
// 	}
	mtickerDo--;
	ticker();
	printf(">>>>>>22222 UDPConn::flushW 1 fd=%d ticker\n", mfd);

	printf(">>>>>>11111 UDPConn::flushW 1 fd=%d leave,mwatcherWriteIO is null %s,waitsnd=%d,snd=%d\n",
		mfd, mwatcherWriteIO ? "false" : "true", ikcp_waitsnd(mkcp), mkcp->snd_wnd);
	return CMS_OK;
}

UdpAddr UDPConn::udpAddr()
{
	return mua;
}

void UDPConn::evWriteIO(cms_net_ev *ev)
{
	atomicInc(ev);		//计数器加1
	mwatcherWriteIO = ev;
}

void UDPConn::pushUM(UdpMsg *um)
{
	mlockUdpMsg.Lock();
	if (mqueueUdpMsg.size() < KCP_QUEUE_LIMIT)
	{		
		mqueueUdpMsg.push(um);
	}
	else
	{
		//数据处理不过来
		delete []um->mdata;
		delete um;
	}
	mlockUdpMsg.Unlock();
}

bool UDPConn::popUM(UdpMsg **um)
{
	mlockUdpMsg.Lock();
	if (mqueueUdpMsg.empty())
	{
		mlockUdpMsg.Unlock();
		return false;
	}
	*um = mqueueUdpMsg.front();
	mqueueUdpMsg.pop();
	mlockUdpMsg.Unlock();
	return true;
}

void UDPConn::ticker()
{	
	if (mtimer == NULL)
	{
		muid = getUdpUid();
		mtimer = mallcoCmsUdpTimer();
		assert(mtimer != NULL);
		pushUDPConn(this);
		cms_udp_timer_init(mtimer, mfd, udpTickerCallBack, muid);
		printf(">>>>>>22222 UDPConn::ticker 1 fd=%d ticker\n", mfd);
	}
	assert(mtickerDo == 0);
	cms_udp_timer_start(mtimer);
	mtickerDo++;
}

//listener
UDPListener::UDPListener()
{
	mruning = false;
	miBindPort = 0;
	mfd = -1;
	mtid = -1;
	mcallBack = NULL;
	msockIO = NULL;
}

UDPListener::~UDPListener()
{

}

void UDPListener::setIO8CallBack(cms_net_ev *sockIO,cms_net_cb cb)
{
	msockIO = sockIO;
	mcallBack = cb;
}

bool UDPListener::run()
{
	int res = cmsCreateThread(&mtid,routinue,this,false);
	if (res == -1)
	{
		char date[128] = {0};
		getTimeStr(date);
		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}

void UDPListener::stop()
{
	logs->info("### UDPListener begin stop listening %s ###",mlistenAddr.c_str());
	if (mruning)
	{
		mruning = false;
		::close(mfd);
		mfd = -1;
	}
	if (mtid > 0)
	{
		cmsWaitForThread(mtid,NULL);
		mtid = -1;
	}	
	logs->info("### UDPListener finish stop listening %s ###",mlistenAddr.c_str());
}

void UDPListener::pushUM(UdpMsg *um)
{
	mlockUdpMsg.Lock();
	mqueueUdpMsg.push(um);
	mlockUdpMsg.Unlock();
}

bool UDPListener::popUM(UdpMsg **um)
{
	mlockUdpMsg.Lock();
	if (mqueueUdpMsg.empty())
	{
		mlockUdpMsg.Unlock();
		return false;
	}
	*um = mqueueUdpMsg.front();
	mqueueUdpMsg.pop();
	mlockUdpMsg.Unlock();
	return true;
}

void UDPListener::thread()
{
	logs->debug("### UDPListener thread=%d ###", gettid());
	bool isSucc;
	UdpMsg *um;
	bool isReadEvent;
	UdpConnInfo *connInfo = NULL;
	while (mruning)
	{
		isSucc = popUM(&um);
		if (isSucc)
		{
			isReadEvent = false;
			mlockUdpConn.Lock();
			MapUdpConnIter it = mmapUdpConn.find(um->mua);
			if (it != mmapUdpConn.end())
			{
				connInfo = it->second;
				connInfo->mconn->pushUM(um);		//udp连接投递数据
				//udp listener 由自己本身投递读事件
				if (connInfo->mwatcherReadIO)
				{
					connInfo->mwatcherReadIO->mcallBack(connInfo->mwatcherReadIO,EventRead);
				}

				if (!mlistUdpConn.empty())
				{
					isReadEvent = true;	//有新的连接没有处理
				}
			}
			else
			{
				//新连接
				//um->mua 处理完会被删除
				UdpAddr &ua = um->mua;
				
				connInfo = new UdpConnInfo;
				connInfo->mconn = new UDPConn(ua,um->mconv,mfd,ua.muiAddr,ua.miPort,true,mlistenType);				
				
				mmapUdpConn.insert(make_pair(ua,connInfo));			//保存udp连接和对应的唯一地址
				connInfo->mconn->pushUM(um);						//udp连接投递数据
				mlistUdpConn.push_back(connInfo->mconn);

				isReadEvent = true;				
			}
			mlockUdpConn.Unlock();
			if (isReadEvent)
			{
				//投递读事件 新的udp连接
				if (mcallBack)
				{
					mcallBack(msockIO,EventRead);
				}
			}
		}
		else
		{
			cmsSleep(10);
		}
	}
}

void *UDPListener::routinue(void *param)
{
	UDPListener *pmgr = (UDPListener *)param;
	pmgr->thread();
	return NULL;
}

int  UDPListener::listen(char* addr,ConnType listenType)
{
	std::string sAddr;
	sAddr.append(addr);
	mlistenAddr = addr;
	struct sockaddr_in serv_addr;
	std::string strHost;
	unsigned short port;
	size_t pos = sAddr.find(":");
	if (pos == std::string::npos)
	{
		logs->error("*** TCPConn dialTcp addr %s is illegal *****",addr);
		return CMS_ERROR;
	}
	strHost = sAddr.substr(0,pos);
	port = (unsigned short)atoi(sAddr.substr(pos+1).c_str());
	logs->info("##### TCPListener listen addr %s #####",addr);
	if (strHost.length() == 0)
	{
		strHost = "0.0.0.0";
	}
	mfd = socket(AF_INET,SOCK_DGRAM,0);
	if (mfd == CMS_INVALID_SOCK)
	{
		logs->error("*** UDPListener listen create socket is error,errno=%d,errstr=%s *****",errno,strerror(errno));
		return CMS_ERROR;
	}
	int n = 1;
	if (setsockopt(mfd, SOL_SOCKET, SO_REUSEADDR, (char *)&n, sizeof(n)) < 0)
	{
		logs->error("*** TCPListener listen set SO_REUSEADDR fail,errno=%d,errstr=%s *****",errno,strerror(errno));
	}
	if (setsockopt(mfd, SOL_SOCKET, SO_REUSEADDR, (char *)&n, sizeof(n)) < 0)
	{
		logs->error("*** UDPListener listen set SO_REUSEADDR fail,errno=%d,errstr=%s *****",errno,strerror(errno));
	}
	nonblocking(mfd);
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	unsigned long ip;
	if (!CDnsCache::instance()->host2ip(strHost.c_str(),ip))
	{
		logs->error("*** UDPListener listen dns cache error *****");
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	serv_addr.sin_addr.s_addr = ip;
	if (bind(mfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		logs->error("*** UDPListener listen bind %s socket is error,errno=%d,errstr=%s *****",mlistenAddr.c_str(),errno,strerror(errno));
		::close(mfd);
		mfd = -1;
		return CMS_ERROR;
	}
	miBindPort = port;
	mruning = true;
	//createThread(this);
	mlistenType = listenType;
	if (mlistenType == TypeHttps)
	{
		if (s2n_init() < 0) 
		{
			logs->error("*** UDPListener error running s2n_init(): '%s' ***", s2n_strerror(s2n_errno, "EN"));
			::close(mfd);
			mfd = -1;
			return CMS_ERROR;
		}
		//还需要设置证书
	}
	return CMS_OK;
}

ConnType UDPListener::listenType()
{
	return mlistenType;
}

bool UDPListener::isTcp()
{
	return false;
}

void *UDPListener::oneConn()
{
	UDPConn* conn = NULL;	
	mlockUdpConn.Lock();
	if (!mlistUdpConn.empty())
	{
		ListUdpConnIter it = mlistUdpConn.begin();
		conn = *it;
		mlistUdpConn.erase(it);
		logs->debug(">>>>UDPListener oneConn.");
	}
	mlockUdpConn.Unlock();	
	return conn;
}

void UDPListener::oneConnRead(void *one,Conn *conn)
{
	logs->debug(">>>>UDPListener oneConnRead.");
	UDPConn* udpConn = (UDPConn*)one;	
	mlockUdpConn.Lock();
	MapUdpConnIter it = mmapUdpConn.find(udpConn->udpAddr());
	if (it != mmapUdpConn.end())
	{
		//读事件
		UdpConnInfo *connInfo = it->second;
		connInfo->mwatcherReadIO = mallcoCmsNetEv();
		initCmsNetEv(connInfo->mwatcherReadIO,readEV,connInfo->mconn->fd(),EventRead);
		conn->evReadIO(connInfo->mwatcherReadIO);
		connInfo->mwatcherReadIO->mcallBack(connInfo->mwatcherReadIO,EventRead);

		//写事件
		cms_net_ev *watcherWriteIO = mallcoCmsNetEv();
		initCmsNetEv(watcherWriteIO,writeEV,connInfo->mconn->fd(),EventWrite);
		connInfo->mconn->evWriteIO(watcherWriteIO);
		watcherWriteIO->mcallBack(watcherWriteIO,EventWrite);
		freeCmsNetEv(watcherWriteIO); //计数器已经在udp中加1 该处需要减1
	}
	else
	{
		logs->warn("######@@@@@ [never] UDPListener oneConn not udpConn @@@@@#####");
		udpConn = NULL;
	}
	mlockUdpConn.Unlock();
}

void UDPListener::delOneConn(UdpAddr ua)
{
	mlockUdpConn.Lock();
	MapUdpConnIter it = mmapUdpConn.find(ua);
	if (it != mmapUdpConn.end())
	{
		freeCmsNetEv(it->second->mwatcherReadIO);
		delete it->second;
		mmapUdpConn.erase(it);
	}
	/*ListUdpConnIter itl = mlistUdpConn.begin();
	for (; itl != mlistUdpConn.end();)
	{
		//理论上不会出现
		logs->warn("#####@@@@@@ should not exist @@@@@@######");
		if ((*itl)->udpAddr() == ua)
		{
			mlistUdpConn.erase(itl);
		}
		else
		{
			itl++;
		}
	}*/
	mlockUdpConn.Unlock();
}

int  UDPListener::accept()
{
	struct sockaddr_in addr;
	socklen_t addrLen = sizeof(addr);
	int ret;
	char *data = NULL;
	IUINT32 conv;
	int len = MTU_LIMIT;
	do
	{
		if (data == NULL)
		{
			data = new char[MTU_LIMIT];
		}
		ret = recvfrom(mfd, data, len, 0, (struct sockaddr*)&addr, &addrLen);
		if (ret >= (int)IKCP_OVERHEAD)
		{
			int r = ikcp_check_legal(data,ret,conv);
			if (r == 0)
			{
				//数据合法 允许处理
				UdpMsg *um = new UdpMsg;
				um->mdata = data;
				um->mlen = ret;
				um->mua.miPort = ntohs(addr.sin_port);
				um->mua.muiAddr = addr.sin_addr.s_addr;
				um->mua.miBindPort = miBindPort;
				um->mua.mlistener = this;
				um->mconv = conv;
				pushUM(um);
				data = NULL;
			}
		}
		else if (ret == -1)
		{
			delete[] data;
			data = NULL;
			break;
		}
	}while (true);
	return 0;
}

int  UDPListener::fd()
{
	return mfd;
}


