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
#include <app/cms_server.h>
#include <config/cms_config.h>
#include <common/cms_utility.h>
#include <log/cms_log.h>
#include <dispatch/cms_net_dispatch.h>

CServer *CServer::minstance = NULL;
CServer::CServer()
{
	mHttp = new TCPListener();
	mHttps = new TCPListener();
	mRtmp = new TCPListener();
	muRtmp = new UDPListener();
	mQuery = new TCPListener();
}

CServer::~CServer()
{
	delete mHttp;
	delete mHttps;
	delete mRtmp;
	delete muRtmp;
	delete mQuery;
}

CServer *CServer::instance()
{
	if (minstance == NULL)
	{
		minstance = new CServer();
	}
	return minstance;
}

void CServer::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}

bool CServer::listenAll()
{
	initUdpSocket();
	if (mHttp->listen(CConfig::instance()->addrHttp()->addr(),TypeHttp) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen http fail *****");
		return false;
	}
	if (CConfig::instance()->certKey()->isOpenSSL() &&
		mHttps->listen(CConfig::instance()->addrHttps()->addr(),TypeHttps) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen https fail *****");
		return false;
	}
	if (mRtmp->listen(CConfig::instance()->addrRtmp()->addr(),TypeRtmp) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen tcp rtmp fail *****");
		return false;
	}
	if (muRtmp->listen(CConfig::instance()->addrRtmp()->addr(),TypeRtmp) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen udp rtmp fail *****");
		return false;
	}
	if (mQuery->listen(CConfig::instance()->addrQuery()->addr(),TypeQuery) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen query fail *****");
		return false;
	}
	//udp 比较特殊
	if (!muRtmp->run())
	{
		logs->error("***** [CServer::listenAll] run udp rtmp fail *****");
		return false;
	}
	//udp监控数据可读 监听sock是否可读还是走正常的网络监听
	cms_net_ev *sockIO = CNetDispatch::instance()->addOneListenDispatch(muRtmp->fd(),muRtmp);
	muRtmp->setIO8CallBack(sockIO,acceptEV);
	//udp 比较特殊 结束
	CNetDispatch::instance()->addOneListenDispatch(mHttp->fd(),mHttp);
	if (CConfig::instance()->certKey()->isOpenSSL())
	{
		CNetDispatch::instance()->addOneListenDispatch(mHttps->fd(),mHttps);
	}	
	CNetDispatch::instance()->addOneListenDispatch(mRtmp->fd(),mRtmp);
	CNetDispatch::instance()->addOneListenDispatch(mQuery->fd(),mQuery);
	return true;
}

