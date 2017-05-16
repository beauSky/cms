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
#include <net/cms_net_mgr.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>

#define VectorVectorNTIneroter std::vector<std::vector<CNetThread *> *>::iterator
#define VectorNTIneroter std::vector<CNetThread *>::iterator

CNetMgr *CNetMgr::minstance = NULL;
CNetMgr::CNetMgr()
{
// 	misRun = false;
// 	mtid = -1;
}

CNetMgr::~CNetMgr()
{

}

CNetMgr *CNetMgr::instance()
{
	if (minstance == NULL)
	{
		minstance = new CNetMgr;
	}
	return minstance;
}

void CNetMgr::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}
// 
// void *CNetMgr::routinue(void *param)
// {
// 	CNetMgr *pmgr = (CNetMgr *)param;
// 	pmgr->thread();
// 	return NULL;
// }
// 
// void CNetMgr::thread()
// {
// 	logs->debug("### CNetMgr thread=%d ###", gettid());
// 	while (misRun)
// 	{
// 		
// 	}
// 	logs->debug("### CNetMgr leave thread=%d ###", gettid());
// }
// 
// bool CNetMgr::run()
// {
// 	if (misRun)
// 	{
// 		return true;
// 	}
// 	misRun = true;
// 	int res = cmsCreateThread(&mtid,routinue,this,false);
// 	if (res == -1)
// 	{
// 		char date[128] = {0};
// 		getTimeStr(date);
// 		logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
// 		return false;
// 	}
// 	return true;
// }
// 
// void CNetMgr::stop()
// {
// 	misRun = false;
// 	cmsWaitForThread(mtid,NULL);
// }

void CNetMgr::cneStart(cms_net_ev *cne,bool isListen/* = false*/)
{
	CNetThread *cnt = NULL;
	uint32 i = (uint32)cne->mfd / MAX_NET_THREAD_FD_NUM;
	mlockNetThread.Lock();
	if (mvnetThread.size() > i)
	{
		cnt = mvnetThread.at(i);
	}
	else
	{		
		for (uint32 n = mvnetThread.size(); n < i; n++)
		{			
			mvnetThread.push_back(NULL);
		}
		cnt = new CNetThread();
		if (!cnt->run())
		{
			logs->error("***** CNetThread run fail *****");
			cmsSleep(1000*3);
			exit(0);
		}
		mvnetThread.push_back(cnt);
	}
	if (cnt == NULL)
	{
		cnt = new CNetThread();
		if (!cnt->run())
		{
			logs->error("***** CNetThread run fail *****");
			cmsSleep(1000*3);
			exit(0);
		}
		mvnetThread[i] = cnt;
	}
	cnt->cneStart(cne,isListen);
	mlockNetThread.Unlock();
}

void CNetMgr::cneStop(cms_net_ev *cne)
{
	CNetThread *cnt = NULL;
	uint32 i = (uint32)cne->mfd / MAX_NET_THREAD_FD_NUM;
	mlockNetThread.Lock();
	if (mvnetThread.size() > i)
	{
		cnt = mvnetThread[i];
		cnt->cneStop(cne);
		if (cnt->cneSize() == 0)
		{
			logs->debug("\n##### begin stop CNetThread %p #####\n",cnt);
			cnt->stop();
			logs->debug("\n##### finish stop CNetThread %p #####\n",cnt);
			delete cnt;
			mvnetThread[i] = NULL;
		}
	}	
	mlockNetThread.Unlock();
}

