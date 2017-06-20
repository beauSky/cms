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
#include <dispatch/cms_net_dispatch.h>
#include <conn/cms_conn_rtmp.h>
#include <conn/cms_conn_mgr.h>
#include <conn/cms_http_s.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <conn/cms_conn_mgr.h>
#include <net/cms_net_mgr.h>
#include <ev/cms_ev.h>
#include <ev/cms_ev.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>

#define MaxDispatchNum 1000
#define VDSPtr vector<CDispatch*>*
#define MapTcpListenIter map<int,TCPListener *>::iterator

CNetDispatch *CNetDispatch::minstance = NULL;
CNetDispatch::CNetDispatch()
{
	
}

CNetDispatch::~CNetDispatch()
{
	
}

CNetDispatch *CNetDispatch::instance()
{
	if (minstance == NULL)
	{
		minstance = new CNetDispatch();
	}
	return minstance;
}

void CNetDispatch::freeInstance()
{
	if (minstance != NULL)
	{
		delete minstance;
		minstance = NULL;
	}
}

void CNetDispatch::addOneDispatch(int fd, CDispatch *ds)
{	
	VDSPtr ptr = NULL;
	mdispatchLock.WLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum 个为一个队列
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
	}
	else
	{		
		for (unsigned int n = mfdDispatch.size(); n < (unsigned int)i; n++)
		{			
			mfdDispatch.push_back(NULL);
		}
		ptr = new vector<CDispatch*>;
		mfdDispatch.push_back(ptr);
	}
	if (ptr == NULL)
	{
		//以前没有开辟过空间
		ptr = new vector<CDispatch*>;
		mfdDispatch[i] = ptr;
	}
	i = fd % MaxDispatchNum; //选中队列中的位置
	for (unsigned int n = ptr->size(); n < (unsigned int)(i+1); n++)  //必须预分配位置
	{
		ptr->push_back(NULL);
	}
	assert(ptr->at(i) == NULL);
	(*ptr)[i] = ds;
	mdispatchLock.UnWLock();
}

void CNetDispatch::delOneDispatch(int fd)
{
	VDSPtr ptr = NULL;
	mdispatchLock.WLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum 个为一个队列
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //选中队列中的位置
		if ((ptr->size() > (unsigned int)i))
		{
			//assert(ptr->at(i) != NULL);
			(*ptr)[i] = NULL;
		}
	}	
	mdispatchLock.UnWLock();
}

void CNetDispatch::dispatchEv(cms_net_ev *watcherRead,cms_net_ev *watcherWrite,int fd,int events)
{
	VDSPtr ptr = NULL;
	mdispatchLock.RLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum 个为一个队列
	assert(mfdDispatch.size() > (unsigned int)i);
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //选中队列中的位置
		if ((ptr->size() > (unsigned int)i) && ptr->at(i))
		{
			//assert(ptr->at(i) != NULL);
			ptr->at(i)->pushEv(fd,events,watcherRead,watcherWrite);
		}
	}	
	mdispatchLock.UnRLock();
}

void CNetDispatch::dispatchEv(cms_timer *ct,int fd,int events)
{
	VDSPtr ptr = NULL;
	mdispatchLock.RLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum 个为一个队列
	assert(mfdDispatch.size() > (unsigned int)i);
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //选中队列中的位置
		if ((ptr->size() > (unsigned int)i) && ptr->at(i))
		{
			//assert(ptr->at(i) != NULL);
			ptr->at(i)->pushEv(fd,events,ct);
		}
		else
		{
			atomicDec(ct);
		}
	}	
	mdispatchLock.UnRLock();
}

void CNetDispatch::addOneListenDispatch(int fd, TCPListener *tls)
{
	mdispatchListenLock.WLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	assert(it == mdispatchListen.end());
	mdispatchListen.insert(make_pair(fd,tls));
	mdispatchListenLock.UnWLock();

	cms_net_ev *sockIO = mallcoCmsNetEv();
	initCmsNetEv(sockIO,acceptEV,fd,EventRead);
	CNetMgr::instance()->cneStart(sockIO,true);
}

void CNetDispatch::delOneListenDispatch(int fd)
{
	mdispatchListenLock.WLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	if (it != mdispatchListen.end())
	{
		it->second->stop();
		delete it->second;
		mdispatchListen.erase(it);
	}
	mdispatchListenLock.UnWLock();
}

void CNetDispatch::dispatchAccept(cms_net_ev *watcher,int fd)
{
	int cfd = CMS_ERROR;
	ConnType listenType = TypeNetNone;
	mdispatchListenLock.RLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	if (it != mdispatchListen.end())
	{
		cfd = it->second->accept();	
		listenType = it->second->listenType();
	}
	mdispatchListenLock.UnRLock();
	if (cfd != CMS_ERROR)
	{
		nonblocking(cfd);
		TCPConn *tcp = new TCPConn(cfd);
		if (listenType == TypeHttp)
		{
			CHttpServer *hs = new CHttpServer(tcp,false);
			if (hs->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,hs);
				hs->evReadIO();
			}
			else
			{
				delete hs;
			}
		}
		else if (listenType == TypeHttps)
		{
			CHttpServer *hs = new CHttpServer(tcp,true);
			if (hs->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,hs);
				hs->evReadIO();
			}
			else
			{
				delete hs;
			}
		}
		else if (listenType == TypeQuery)
		{
			CHttpServer *hs = new CHttpServer(tcp,false);
			if (hs->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,hs);
				hs->evReadIO();
			}
			else
			{
				delete hs;
			}
		}
		else if (listenType == TypeRtmp)
		{
			CConnRtmp *rtmp = new CConnRtmp(RtmpServerBPlayOrPublish,tcp,"","");
			if (rtmp->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,rtmp);
				rtmp->evReadIO();
			}
			else
			{
				delete rtmp;
			}
		}
	}
}


