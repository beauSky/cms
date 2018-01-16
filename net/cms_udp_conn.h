/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: Ìì¿ÕÃ»ÓÐÎÚÔÆ/kisslovecsh@foxmail.com

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
#ifndef __CMS_UDP_CONN_H__
#define __CMS_UDP_CONN_H__
#include <interface/cms_read_write.h>
#include <interface/cms_conn_listener.h>
#include <kcp/ikcp.h>
#include <core/cms_thread.h>
#include <common/cms_type.h>
#include <net/cms_net_var.h>
#include <net/cms_udp_timer.h>
#include <core/cms_lock.h>
#include <netinet/in.h>
#include <string>
#include <queue>
#include <list>
#include <map>
using namespace std;

#define	MTU_LIMIT			2048
#define KCP_QUEUE_LIMIT		1024

typedef struct _UdpMsg{
	UdpAddr mua;
	char	*mdata;
	int		mlen;
	IUINT32 mconv;
}UdpMsg;

void initUdpSocket();

class UDPConn:public CReaderWriter
{
public:
	UDPConn(UdpAddr ua,IUINT32 conv,int fd,unsigned long ipInt,unsigned short port,bool isListen,ConnType connectType);
	UDPConn();
	~UDPConn();

	int   dialUdp(char *addr,ConnType connectType);
	int	  connect();
	int   read(char* dstBuf,int len,int &nread);
	int   write(char *srcBuf,int len,int &nwrite);
	void  close();
	int   fd();
	int   remoteAddr(char *addr,int len);
	int   localAddr(char *addr,int len);
	void  setReadTimeout(long long readTimeout);
	long long getReadTimeout();
	void  setWriteTimeout(long long writeTimeout);
	long long getWriteTimeout();
	long long getReadBytes();
	long long getWriteBytes();	
	ConnType connectType();
	char  *errnoCode();
	int   errnos();
	int   setNodelay(int on);
	int	  setReadBuffer(int size);
	int	  setWriteBuffer(int size);
	int   flushR();
	int   flushW();
	UdpAddr udpAddr();
	void  recvData();

	cms_net_ev *evWriteIO();
	cms_net_ev *evReadIO();

	void  pushUM(UdpMsg *um);
	void  ticker();
	bool  isClose();
	uint32 getCloseTime();
	bool  isUid(uint64 uid);
private:
	bool  popUM(UdpMsg **um);
	

	cms_udp_timer *mtimer;
	uint64		  muid;
	int           mtickerDo;
	bool		  misClose;
	uint32		  mcloseTime;
	bool		  misListen;
	int			  mlsFd;
	int			  mfd;	
	UdpAddr		  mua;
	struct sockaddr_in mto;
	string		  mraddr;
	string		  mladdr;
	long long	  mreadTimeout;
	long long	  mreadBytes;
	long long	  mwriteTimetou;
	long long	  mwriteBytes;
	ConnType	  mconnectType;
	int			  merrcode;	

	ikcpcb		  *mkcp;
	CLock		  mlockKcp;

	queue<UdpMsg*>	mqueueUdpMsg;
	CLock			mlockUdpMsg;

	cms_net_ev		*mwatcherWriteIO;	//Ð´Çý¶¯Æ÷
	cms_net_ev		*mwatcherReadIO;	//¶ÁÇý¶¯Æ÷

	//test
	int mwriteTotalLen;
};

typedef struct _UdpConnInfo{
public:
	UDPConn     *mconn;

	_UdpConnInfo()
	{
		mconn = NULL;
	}
}UdpConnInfo;


class UDPListener:public CConnListener
{
public:
	UDPListener();
	~UDPListener();

	bool run();
	void thread();
	static void *routinue(void *param);
	void setIO8CallBack(cms_net_ev *sockIO,cms_net_cb cb);
	void delOneConn(UdpAddr ua);
	//ÖØÔØ
	int  listen(char* addr,ConnType listenType);
	void stop();
	int  accept();
	int  fd();
	ConnType listenType();
	bool isTcp();
	void *oneConn();
private:
	void pushUM(UdpMsg *um);
	bool popUM(UdpMsg **um);

	cms_net_cb	mcallBack;
	cms_net_ev *msockIO;

	string  mlistenAddr;
	int		miBindPort;
	bool	mruning;
	int     mfd;
	ConnType mlistenType;

	cms_thread_t mtid;

	list<UDPConn *>				mlistUdpConn;
	map<UdpAddr,UdpConnInfo*>	mmapUdpConn;
	CLock						mlockUdpConn;

	queue<UdpMsg*>	mqueueUdpMsg;
	CLock			mlockUdpMsg;
};
#endif
