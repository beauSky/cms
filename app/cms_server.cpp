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
	mQuery = new TCPListener();
}

CServer::~CServer()
{
	delete mHttp;
	delete mHttps;
	delete mRtmp;
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

bool	CServer::listenAll()
{
	if (mHttp->listen(CConfig::instance()->addrHttp()->addr(),TypeHttp) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen http fail *****");
		return false;
	}
	if (mHttps->listen(CConfig::instance()->addrHttps()->addr(),TypeHttps) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen https fail *****");
		return false;
	}
	if (mRtmp->listen(CConfig::instance()->addrRtmp()->addr(),TypeRtmp) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen rtmp fail *****");
		return false;
	}
	if (mQuery->listen(CConfig::instance()->addrQuery()->addr(),TypeQuery) == CMS_ERROR)
	{
		logs->error("***** [CServer::listenAll] listen query fail *****");
		return false;
	}
	CNetDispatch::instance()->addOneListenDispatch(mHttp->fd(),mHttp);
	CNetDispatch::instance()->addOneListenDispatch(mHttps->fd(),mHttps);
	CNetDispatch::instance()->addOneListenDispatch(mRtmp->fd(),mRtmp);
	CNetDispatch::instance()->addOneListenDispatch(mQuery->fd(),mQuery);
	return true;
}

