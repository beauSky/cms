#include <taskmgr/cms_task_mgr.h>
#include <net/cms_tcp_conn.h>
#include <conn/cms_conn_mgr.h>
#include <log/cms_log.h>
#include <assert.h>

#define MapTaskConnIteror std::map<HASH,Conn *>::iterator

CTaskMgr *CTaskMgr::minstance = NULL;
CTaskMgr::CTaskMgr()
{
	misRun = false;
}

CTaskMgr::~CTaskMgr()
{

}

CTaskMgr *CTaskMgr::instance()
{
	if (minstance == NULL)
	{
		minstance = new CTaskMgr;
	}
	return minstance;
}

void  CTaskMgr::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}

void *CTaskMgr::routinue(void *param)
{
	CTaskMgr *pIns = (CTaskMgr*)param;
	pIns->thread();
	return NULL;
}

void CTaskMgr::thread()
{
	logs->info(">>>>> CTaskMgr thread pid=%d\n",gettid());
	CreateTaskPacket *ctp;
	bool isPop;
	while (misRun)
	{
		isPop = pop(&ctp);
		if (isPop)
		{
			if (ctp->createAct == CREATE_ACT_PULL)
			{
				pullCreateTask(ctp);
			}
			else if (ctp->createAct == CREATE_ACT_PUSH)
			{
				pushCreateTask(ctp);
			}
			delete ctp;
		}
		else
		{
			cmsSleep(10);
		}
	}
	logs->info(">>>>> CTaskMgr thread leave pid=%d\n",gettid());
}

bool CTaskMgr::run()
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

void CTaskMgr::createTask(std::string pullUrl,std::string pushUrl,std::string refer,
				   int createAct,bool isHotPush,bool isPush2Cdn)
{
	CreateTaskPacket * ctp = new CreateTaskPacket;
	ctp->createAct = createAct;
	ctp->pullUrl = pullUrl;
	ctp->pushUrl = pushUrl;
	ctp->refer = refer;
	ctp->isHotPush = isHotPush;
	ctp->isPush2Cdn = isPush2Cdn;
	ctp->ID = 0;
	push(ctp);
}

void CTaskMgr::push(CreateTaskPacket *ctp)
{
	mlockQueue.Lock();
	mqueueCTP.push(ctp);
	mlockQueue.Unlock();
}

bool CTaskMgr::pop(CreateTaskPacket **ctp)
{
	bool isPop = false;
	mlockQueue.Lock();
	if (!mqueueCTP.empty())
	{
		isPop = true;
		*ctp = mqueueCTP.front();
		mqueueCTP.pop();
	}
	mlockQueue.Unlock();
	return isPop;
}

void CTaskMgr::pullCreateTask(CreateTaskPacket *ctp)
{
	assert(ctp != NULL);
	LinkUrl linUrl;
	if (parseUrl(ctp->pullUrl,linUrl))
	{
		CConnMgrInterface::instance()->createConn((char *)linUrl.addr.c_str(),ctp->pullUrl,"",TypeRtmp,RtmpClient2Play);
	}
	else
	{
		logs->error("*** [CTaskMgr::pullCreateTask] %s parse pull url fail ***",ctp->pullUrl.c_str());
	}	
}

void CTaskMgr::pushCreateTask(CreateTaskPacket *ctp)
{
	assert(ctp != NULL);
	LinkUrl linUrl;
	if (parseUrl(ctp->pushUrl,linUrl))
	{
		CConnMgrInterface::instance()->createConn((char *)linUrl.addr.c_str(),ctp->pullUrl,ctp->pushUrl,TypeRtmp,RtmpClient2Publish);
	}
	else
	{
		logs->error("*** [CTaskMgr::pushCreateTask] %s parse push url fail ***",ctp->pushUrl.c_str());
	}	
}

bool CTaskMgr::pullTaskAdd(HASH &hash,Conn *conn)
{
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it = mmapPullTaskConn.find(hash);
	if (it != mmapPullTaskConn.end())
	{
		mlockPullTaskConn.Unlock();
		return false;
	}
	mmapPullTaskConn[hash] = conn;
	mlockPullTaskConn.Unlock();
	return true;
}

bool CTaskMgr::pullTaskDel(HASH &hash)
{
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it = mmapPullTaskConn.find(hash);
	if (it == mmapPullTaskConn.end())
	{
		mlockPullTaskConn.Unlock();
		return false;
	}
	mmapPullTaskConn.erase(it);
	mlockPullTaskConn.Unlock();
	return true;
}

bool CTaskMgr::pullTaskStop(HASH &hash)
{
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it = mmapPullTaskConn.find(hash);
	if (it != mmapPullTaskConn.end())
	{
		mlockPullTaskConn.Unlock();
		return false;
	}
	it->second->stop("<<stop task by pullTaskStop func>>");
	mlockPullTaskConn.Unlock();
	return true;
}

void CTaskMgr::pullTaskStopAll()
{
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it;
	for (it = mmapPullTaskConn.begin(); it != mmapPullTaskConn.end(); it++ )
	{
		it->second->stop("<<stop task by pullTaskStopAll func>>");
	}	
	mlockPullTaskConn.Unlock();
}

void CTaskMgr::pullTaskStopAllByIP(std::string strIP)
{
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it;
	for (it = mmapPullTaskConn.begin(); it != mmapPullTaskConn.end(); it++ )
	{
		if (it->second->getRemoteIP() == strIP)
		{
			it->second->stop("<<stop task by pullTaskStopAllByIP func>>");
		}
	}	
	mlockPullTaskConn.Unlock();
}

bool CTaskMgr::pullTaskIsExist(HASH &hash)
{
	bool isExist = false;
	mlockPullTaskConn.Lock();
	MapTaskConnIteror it = mmapPullTaskConn.find(hash);
	if (it != mmapPullTaskConn.end())
	{
		isExist = true;
	}
	mlockPullTaskConn.Unlock();
	return isExist;
}
//·ÖË®Áë
bool CTaskMgr::pushTaskAdd(HASH &hash,Conn *conn)
{
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it = mmapPushTaskConn.find(hash);
	if (it != mmapPushTaskConn.end())
	{
		mlockPushTaskConn.Unlock();
		return false;
	}
	mmapPushTaskConn[hash] = conn;
	mlockPushTaskConn.Unlock();
	return true;
}

bool CTaskMgr::pushTaskDel(HASH &hash)
{
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it = mmapPushTaskConn.find(hash);
	if (it == mmapPushTaskConn.end())
	{
		mlockPushTaskConn.Unlock();
		return false;
	}
	mmapPushTaskConn.erase(it);
	mlockPushTaskConn.Unlock();
	return true;
}

bool CTaskMgr::pushTaskStop(HASH &hash)
{
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it = mmapPushTaskConn.find(hash);
	if (it != mmapPushTaskConn.end())
	{
		mlockPushTaskConn.Unlock();
		return false;
	}
	it->second->stop("<<stop task by pushTaskStop func>>");
	mlockPushTaskConn.Unlock();
	return true;
}

void CTaskMgr::pushTaskStopAll()
{
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it;
	for (it = mmapPushTaskConn.begin(); it != mmapPushTaskConn.end(); it++ )
	{
		it->second->stop("<<stop task by pushTaskStopAll func>>");
	}	
	mlockPushTaskConn.Unlock();
}

void CTaskMgr::pushTaskStopAllByIP(std::string strIP)
{
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it;
	for (it = mmapPushTaskConn.begin(); it != mmapPushTaskConn.end(); it++ )
	{
		if (it->second->getRemoteIP() == strIP)
		{
			it->second->stop("<<stop task by pushTaskStopAllByIP func>>");
		}
	}	
	mlockPushTaskConn.Unlock();
}

bool CTaskMgr::pushTaskIsExist(HASH &hash)
{
	bool isExist = false;
	mlockPushTaskConn.Lock();
	MapTaskConnIteror it = mmapPushTaskConn.find(hash);
	if (it != mmapPushTaskConn.end())
	{
		isExist = true;
	}
	mlockPushTaskConn.Unlock();
	return isExist;
}
