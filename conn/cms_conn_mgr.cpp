/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: 天空没有乌云/kisslovecsh@foxmail.com

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
#include <conn/cms_conn_mgr.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <dispatch/cms_net_dispatch.h>
#include <ev/cms_ev.h>
#include <conn/cms_http_c.h>
#include <errno.h>
#include <assert.h>


#define MapConnInter map<int,Conn *>::iterator

CConnMgr::CConnMgr(int i)
{
	mthreadIdx = i;
	mtid = 0;
}

CConnMgr::~CConnMgr()
{

}

void CConnMgr::addOneConn(int fd,Conn *c)
{
	bool isSucc = false;
	mfdConnLock.WLock();
	MapConnInter it = mfdConn.find(fd);
	assert(it == mfdConn.end());
	if (it == mfdConn.end())
	{
		mfdConn.insert(make_pair(fd,c));
		isSucc = true;
	}
	mfdConnLock.UnWLock();

	if (isSucc)
	{
		CNetDispatch::instance()->addOneDispatch(fd,this);
	}	
}

void CConnMgr::delOneConn(int fd)
{
	CNetDispatch::instance()->delOneDispatch(fd);
	Conn *conn = NULL;
	mfdConnLock.WLock();
	MapConnInter it = mfdConn.find(fd);
	if (it != mfdConn.end())
	{
		mfdConn.erase(it);
		conn = it->second;		
	}
	mfdConnLock.UnWLock();
	if (conn)
	{
		delete conn;
	}
}

//TEST
std::map<unsigned long,unsigned long> gmapCSendTakeTime;
int64 gCSendTakeTimeTT;
CLock gCSendTakeTime;
//TEST end

void CConnMgr::dispatchEv(FdEvents *fe)
{
	unsigned long tB = getTickCount();
	bool isSucc = true;
	mfdConnLock.RLock();
	MapConnInter it = mfdConn.find(fe->fd);
	if (it != mfdConn.end())
	{
		int ret = it->second->handleEv(fe);
		if (ret != CMS_OK)
		{
			logs->info("*** [CConnMgr::dispatchEv] one conn %s handle event fail ***",it->second->getUrl().c_str());
			isSucc = false;
			it->second->stop("");
		}
	}
	else
	{
		logs->debug(">>>>CConnMgr::dispatchEv not find sock %d.",fe->fd);
	}
	mfdConnLock.UnRLock();
	if (!isSucc)
	{
		delOneConn(fe->fd);
	}

	unsigned long tE = getTickCount();
	gCSendTakeTime.Lock();
	int64 sendTakeTimeTT = getTimeUnix();
	bool isPrintf = false;
	if (sendTakeTimeTT - gCSendTakeTimeTT > 60)
	{
		isPrintf = true;
		gCSendTakeTimeTT = sendTakeTimeTT;
	}
	printTakeTime(gmapCSendTakeTime,tB,tE,(char *)"CConnMgr",isPrintf);
	gCSendTakeTime.Unlock();
}

void  CConnMgr::pushEv(int fd,int events,cms_net_ev *watcherRead,cms_net_ev *watcherWrite)
{
	FdEvents * fe = new FdEvents;
	fe->fd = fd;
	fe->events = events;
	fe->watcherReadIO = watcherRead;
	fe->watcherWriteIO = watcherWrite;
	fe->watcherWCmsTimer = NULL;
	fe->watcherRCmsTimer = NULL;
	mqueueLock.Lock();
	mqueue.push(fe);
	mqueueLock.Unlock();
}

void CConnMgr::pushEv(int fd,int events,cms_timer *ct)
{
	FdEvents * fe = new FdEvents;
	fe->fd = fd;
	fe->events = events;
	fe->watcherReadIO = NULL;
	fe->watcherWriteIO = NULL;
	if (events == EventWait2Read)
	{
		fe->watcherWCmsTimer = NULL;
		fe->watcherRCmsTimer = ct;
	}
	else if (events == EventWait2Write)
	{
		fe->watcherWCmsTimer = ct;
		fe->watcherRCmsTimer = NULL;
	}
	mqueueLock.Lock();
	mqueue.push(fe);
	mqueueLock.Unlock();
}

bool CConnMgr::popEv(FdEvents **fe)
{
	mqueueLock.Lock();
	if (mqueue.empty())
	{
		mqueueLock.Unlock();
		return false;
	}
	*fe = mqueue.front();
	mqueue.pop();
	mqueueLock.Unlock();
	return true;
}

bool CConnMgr::run()
{
	misRun = true;
	int res = cmsCreateThread(&mtid,routinue,this,false);
	if (res == -1)
	{
		char date[128] = {0};
		getTimeStr(date);
		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}

void CConnMgr::stop()
{
	logs->debug("##### CConnMgr::stop begin #####");
	misRun = false;
	cmsWaitForThread(mtid,NULL);
	mtid = 0;
	logs->debug("##### CConnMgr::stop finish #####");
}

void *CConnMgr::routinue(void *param)
{
	CConnMgr *pmgr = (CConnMgr *)param;
	pmgr->thread();
	return NULL;
}

void CConnMgr::thread()
{
	logs->debug("### CConnMgr thread=%d ###", gettid());

	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(mthreadIdx, &mask);
	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
	{
		logs->error("*****[CConnMgr::thread] set thread affinity failed,err=%d,errstring=%s *****",errno,strerror(errno));
	}

	FdEvents *fe;
	while (misRun)
	{
		fe = NULL;
		if (popEv(&fe))
		{
			assert(fe != NULL);
			dispatchEv(fe);
			if (fe->watcherWCmsTimer)
			{
				//如果不为空，加数器需要减1
				atomicDec(fe->watcherWCmsTimer);
			}
			if (fe->watcherRCmsTimer)
			{
				//如果不为空，加数器需要减1
				atomicDec(fe->watcherRCmsTimer);
			}
			delete fe;
		}
		else
		{
			cmsSleep(30);
		}
	}
	logs->debug("### CConnMgr leave thread=%d ###", gettid());
}

CConnMgrInterface *CConnMgrInterface::minstance = NULL;
CConnMgrInterface::CConnMgrInterface()
{
	misRun = false;
	mtid = 0;
	int num = sysconf(_SC_NPROCESSORS_CONF);
	logs->debug("######## system has %d processor(s) #########", num);
	if (num < NUM_OF_THE_CONN_MGR)
	{
		num = NUM_OF_THE_CONN_MGR;
	}
	for (int i =0; i < num;i ++)
	{
		mconnMgrArray[i] = new CConnMgr(i%num);
		assert(mconnMgrArray[i]->run());
	}
}

CConnMgrInterface::~CConnMgrInterface()
{
	
}

CConnMgrInterface *CConnMgrInterface::instance()
{
	if (minstance == NULL)
	{
		minstance = new CConnMgrInterface();
	}
	return minstance;
}

void CConnMgrInterface::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}

void CConnMgrInterface::addOneConn(int fd,Conn *c)
{
	int i = 0;
	if (fd < 0)
	{
		i = (~(fd-1)) % NUM_OF_THE_CONN_MGR;
	}
	else
	{
		i = fd % NUM_OF_THE_CONN_MGR;
	}
	mconnMgrArray[i]->addOneConn(fd,c);
}

void CConnMgrInterface::delOneConn(int fd)
{
	int i = 0;
	if (fd < 0)
	{
		i = (~(fd-1)) % NUM_OF_THE_CONN_MGR;
	}
	else
	{
		i = fd % NUM_OF_THE_CONN_MGR;
	}
	mconnMgrArray[i]->delOneConn(fd);
}

Conn *CConnMgrInterface::createConn(HASH &hash,char *addr,string pullUrl,std::string pushUrl,std::string oriUrl,std::string strReferer
									,ConnType connectType,RtmpType rtmpType,bool isTcp/* = true*/)
{
	Conn *conn = NULL;
	if (isTcp)
	{
		TCPConn *tcp = new TCPConn();
		if (tcp->dialTcp(addr,connectType) == CMS_ERROR)
		{
			return NULL;
		}
		if (connectType == TypeHttp || connectType == TypeHttps)
		{
			ChttpClient *http = new ChttpClient(hash,tcp,pullUrl,oriUrl,strReferer,connectType == TypeHttp?false:true);
			if (http->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(tcp->fd(),http);

				http->evReadIO();
				http->evWriteIO();
				conn = http;
				if (tcp->connect() == CMS_ERROR)
				{
					CConnMgrInterface::instance()->delOneConn(tcp->fd());
					delete http;
					conn = NULL;
				}
			}
			else
			{
				delete http;
			}
		}
		else if (connectType == TypeRtmp)
		{
			CConnRtmp *rtmp = new CConnRtmp(hash,rtmpType,tcp,pullUrl,pushUrl);
			if (rtmp->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(tcp->fd(),rtmp);

				rtmp->evReadIO();
				rtmp->evWriteIO();
				conn = rtmp;
				if (tcp->connect() == CMS_ERROR)
				{
					CConnMgrInterface::instance()->delOneConn(tcp->fd());
					delete rtmp;
					conn = NULL;
				}
			}
			else
			{
				delete rtmp;
			}
		}
	}
	else
	{
		UDPConn *udp = new UDPConn();
		if (udp->dialUdp(addr,connectType) == CMS_ERROR)
		{
			return NULL;
		}
		if (connectType == TypeRtmp)
		{
			CConnRtmp *rtmp = new CConnRtmp(hash,rtmpType,udp,pullUrl,pushUrl);
			if (rtmp->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(udp->fd(),rtmp);
				rtmp->evWriteIO(udp->evWriteIO());
				rtmp->evReadIO(udp->evReadIO());
				conn = rtmp;
				if (udp->connect() == CMS_ERROR)
				{
					CConnMgrInterface::instance()->delOneConn(udp->fd());
					delete rtmp;
					conn = NULL;
				}
			}
			else
			{
				delete rtmp;
			}
		}
	}
	return conn;
}

void *CConnMgrInterface::routinue(void *param)
{
	CConnMgrInterface *pIns = (CConnMgrInterface*)param;
	pIns->thread();
	return NULL;
}

void CConnMgrInterface::thread()
{
	logs->info(">>>>> CConnMgrInterface thread pid=%d",gettid());
	logs->info(">>>>> CConnMgrInterface thread leave pid=%d",gettid());
}

bool CConnMgrInterface::run()
{	
	if (gCSendTakeTimeTT == 0)
	{
		gCSendTakeTimeTT = getTimeUnix();
	}
	misRun = true;
	int res = cmsCreateThread(&mtid,routinue,this,false);
	if (res == -1)
	{
		char date[128] = {0};
		getTimeStr(date);
		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}

void CConnMgrInterface::stop()
{
	logs->debug("##### CConnMgrInterface::stop begin #####");
	misRun = false;
	cmsWaitForThread(mtid, NULL);
	mtid = 0;
	for (int i = 0; i < NUM_OF_THE_CONN_MGR; i++)
	{
		mconnMgrArray[i]->stop();
		delete mconnMgrArray[i];
		mconnMgrArray[i] = NULL;
	}
	logs->debug("##### CConnMgrInterface::stop finish #####");
}
