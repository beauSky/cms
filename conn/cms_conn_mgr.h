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
#ifndef __CMS_CONN_MGR_H__
#define __CMS_CONN_MGR_H__
#include <interface/cms_dispatch.h>
#include <conn/cms_conn_rtmp.h>
#include <core/cms_lock.h>
#include <core/cms_thread.h>
#include <net/cms_tcp_conn.h>
#include <net/cms_udp_conn.h>
#include <common/cms_var.h>
#include <map>
#include <queue>
using namespace std;

#define NUM_OF_THE_CONN_MGR 16

class CConnMgr:CDispatch
{
public:
	CConnMgr(int i);
	~CConnMgr();

	bool run();
	void stop();
	void thread();
	static void *routinue(void *param);	

	void addOneConn(int fd,Conn *c);
	void delOneConn(int fd);
	void  pushEv(int fd,int events,cms_net_ev *watcherRead,cms_net_ev *watcherWrite);
	void  pushEv(int fd,int events,cms_timer *ct);
private:	
	void dispatchEv(FdEvents *fe);
	bool popEv(FdEvents **fe);	
	map<int,Conn *> mfdConn;
	CRWlock mfdConnLock;

	bool misRun;
	int  mthreadIdx;
	cms_thread_t mtid;
	queue<FdEvents *> mqueue;
	CLock mqueueLock;
};

class CConnMgrInterface
{
public:
	CConnMgrInterface();
	~CConnMgrInterface();

	static CConnMgrInterface *instance();
	static void freeInstance();

	static void *routinue(void *param);
	void thread();
	bool run();
	void stop();

	void addOneConn(int fd,Conn *c);
	void delOneConn(int fd);
	Conn *createConn(HASH &hash,char *addr,string pullUrl,std::string pushUrl,std::string oriUrl,std::string strReferer
		,ConnType connectType,RtmpType rtmpType,bool isTcp = true);
private:
	static CConnMgrInterface *minstance;
	CConnMgr *mconnMgrArray[NUM_OF_THE_CONN_MGR];
	bool			misRun;
	cms_thread_t	mtid;
};
#endif
