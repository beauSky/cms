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
#include <conn/cms_conn_mgr.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <dispatch/cms_net_dispatch.h>
#include <ev/cms_ev.h>
#include <conn/cms_http_c.h>
#include <assert.h>


#define MapConnInter map<int,Conn *>::iterator

void CConnMgr::addOneConn(int fd,Conn *c)
{
	mfdConnLock.WLock();
	MapConnInter it = mfdConn.find(fd);
	assert(it == mfdConn.end());
	if (it == mfdConn.end())
	{
		mfdConn.insert(make_pair(fd,c));
	}
	mfdConnLock.UnWLock();

	CNetDispatch::instance()->addOneDispatch(fd,this);
}

void CConnMgr::delOneConn(int fd)
{
	mfdConnLock.WLock();
	MapConnInter it = mfdConn.find(fd);
	if (it != mfdConn.end())
	{
		mfdConn.erase(it);
		if (it->second)
		{
			delete it->second;
		}
	}
	mfdConnLock.UnWLock();

	CNetDispatch::instance()->delOneDispatch(fd);
}

void CConnMgr::dispatchEv(FdEvents *fe)
{
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
	mfdConnLock.UnRLock();
	if (!isSucc)
	{
		delOneConn(fe->fd);
	}
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
		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}

void CConnMgr::stop()
{
	misRun = false;
	cmsWaitForThread(mtid,NULL);
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
	for (int i =0; i < NUM_OF_THE_CONN_MGR;i ++)
	{
		mconnMgrArray[i] = new CConnMgr();
		assert(mconnMgrArray[i]->run());
	}
}

CConnMgrInterface::~CConnMgrInterface()
{
	for (int i =0; i < NUM_OF_THE_CONN_MGR;i ++)
	{
		mconnMgrArray[i]->stop();
		delete mconnMgrArray[i];
		mconnMgrArray[i] = NULL;
	}
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
	int i = fd % NUM_OF_THE_CONN_MGR;
	mconnMgrArray[i]->addOneConn(fd,c);
}

void CConnMgrInterface::delOneConn(int fd)
{
	int i = fd % NUM_OF_THE_CONN_MGR;
	mconnMgrArray[i]->delOneConn(fd);
}

Conn *CConnMgrInterface::createConn(char *addr,string pullUrl,std::string pushUrl,std::string oriUrl,std::string strReferer
									,ConnType connectType,RtmpType rtmpType)
{
	Conn *conn = NULL;
	TCPConn *tcp = new TCPConn();
	if (tcp->dialTcp(addr,connectType) == CMS_ERROR)
	{
		return NULL;
	}
	if (connectType == TypeHttp || connectType == TypeHttps)
	{
		ChttpClient *http = new ChttpClient(tcp,pullUrl,oriUrl,strReferer,connectType == TypeHttp?false:true);
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
		CConnRtmp *rtmp = new CConnRtmp(rtmpType,tcp,pullUrl,pushUrl);
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
	logs->info(">>>>> CConnMgrInterface thread pid=%d\n",gettid());
	logs->info(">>>>> CConnMgrInterface thread leave pid=%d\n",gettid());
}

bool CConnMgrInterface::run()
{	
	misRun = true;
	int res = cmsCreateThread(&mtid,routinue,this,true);
	if (res == -1)
	{
		char date[128] = {0};
		getTimeStr(date);
		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}