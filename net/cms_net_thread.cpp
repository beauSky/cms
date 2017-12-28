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
#include <net/cms_net_thread.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#define EPOLL_EVENTS_COUNTS 1000
#define VectorIterator std::vector<cms_net_ev *>::iterator

CNetThread::CNetThread()
{
	misRun = false;
	mtid = -1;
	mepfd = -1;
	mcneNum = 0;
	for (int i = 0; i < MAX_NET_THREAD_FD_NUM; i++)
	{
		mvRCNE.push_back(NULL);
		mvWCNE.push_back(NULL);
	}
}

CNetThread::~CNetThread()
{
// 	printf(">>>>>CNetThread::~CNetThread %p .\n",this);
	if (mepfd > 0)
	{
		close(mepfd);
	}
}

void *CNetThread::routinue(void *param)
{
	CNetThread *pthread = (CNetThread *)param;
	pthread->thread();
	return NULL;
}

void CNetThread::thread()
{
	logs->debug("### CNetThread thread=%d ###", gettid());
	int nfds;
	int evs;
	cms_net_ev *cne;
	struct epoll_event epeventsRemote[EPOLL_EVENTS_COUNTS];
	while (misRun)
	{
		nfds = epoll_wait(mepfd,epeventsRemote,EPOLL_EVENTS_COUNTS,100);
		for (int i = 0; i < nfds; ++i)
		{
			evs = 0;			
			if (epeventsRemote[i].events & EPOLLIN)
			{
				evs = EventRead;
				cne = getReadCne(epeventsRemote[i].data.fd);
				if (cne == NULL)
				{
					//可能刚刚被删除了
					continue;
				}
				if (cne)
				{
					cne->mcallBack(cne,evs); //不能阻塞
					atomicDec(cne); //++
				}
			}
			if (epeventsRemote[i].events & EPOLLOUT)
			{
				evs = EventWrite;
				cne = getWriteCne(epeventsRemote[i].data.fd);
				if (cne == NULL)
				{
					//可能刚刚被删除了
					continue;
				}
				if (cne)
				{
					char szTime[30] = { 0 };
					getTimeStr(szTime);
					printf(">>>>>>>>11111 %s CNetThread write event fd=%d\n", szTime, cne->mfd);
					cne->mcallBack(cne,evs); //不能阻塞
					atomicDec(cne); //++
				}
			}
// 			if (epeventsRemote[i].events & EPOLLERR)
// 			{
// 				evs |= EventErrot;
// 			}
			
		}
	}
	logs->debug("### CNetThread leave thread=%d ###", gettid());
}

bool CNetThread::run()
{
	if (misRun)
	{
		return true;
	}
	mepfd = epoll_create(8192);
	if (mepfd == -1)
	{
		logs->error("***** file=%s,line=%d epoll_create fail,errno=%d *****\n",__FILE__,__LINE__,errno);
		return false;
	}
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

void CNetThread::stop()
{
	misRun = false;
	cmsWaitForThread(mtid,NULL);
}

int CNetThread::vectorIdx(int fd)
{
	return fd % MAX_NET_THREAD_FD_NUM;
}

cms_net_ev *CNetThread::getWriteCne(int fd)
{
	int i = vectorIdx(fd);
	assert(i >= 0 && i < MAX_NET_THREAD_FD_NUM);
	mlockCNE.Lock();
	cms_net_ev *cne = mvWCNE[i];
	atomicInc(cne); //++
	mlockCNE.Unlock();
	return cne;
}

cms_net_ev *CNetThread::getReadCne(int fd)
{
	int i = vectorIdx(fd);
	assert(i >= 0 && i < MAX_NET_THREAD_FD_NUM);
	mlockCNE.Lock();
	cms_net_ev *cne = mvRCNE[i];
	atomicInc(cne); //++
	mlockCNE.Unlock();
	return cne;
}

int CNetThread::epollEV(int evs,bool isListen)
{
	int epollEv = 0;
	if (evs & EventRead)
	{
		epollEv |= EPOLLIN;
	}
	if (evs & EventWrite)
	{
		epollEv |= EPOLLOUT;
	}
	if (isListen)
	{
		//epollEv |= EPOLLLT;
	}
	else
	{
		epollEv |= EPOLLET;
	}
	return epollEv;
}

bool CNetThread::isReadEv(int evs)
{
	if (evs & EventRead)
	{
		return true;
	}
	return false;
}

bool CNetThread::isWriteEv(int evs)
{
	if (evs & EventWrite)
	{
		return true;
	}
	return false;
}

void CNetThread::cneStart(cms_net_ev *cne,bool isListen/* = false*/)
{
	bool isExist = false;
	struct epoll_event ev;
	ev.data.fd = cne->mfd;
	ev.events = epollEV(cne->mwatchEvent,isListen);
	int i = vectorIdx(cne->mfd);
	mlockCNE.Lock();
	cms_net_ev *cneoR = NULL;
	cms_net_ev *cneoW = NULL;
	if (isReadEv(cne->mwatchEvent))
	{
		cneoR = mvRCNE[i];
		//可能以前投递过写事件
		if (mvWCNE[i] != NULL)
		{
			ev.events |= epollEV(mvWCNE[i]->mwatchEvent,isListen);
			isExist = true;
		}
	}
	if (isWriteEv(cne->mwatchEvent))
	{
		cneoW = mvWCNE[i];
		//可能以前投递过读事件
		if (mvRCNE[i] != NULL)
		{
			ev.events |= epollEV(mvRCNE[i]->mwatchEvent,isListen);
			isExist = true;
		}
	}
	assert(cneoR == NULL && cneoW == NULL);
	if (isExist)
	{
		epoll_ctl(mepfd,EPOLL_CTL_MOD,cne->mfd,&ev);
	}
	else
	{
		epoll_ctl(mepfd,EPOLL_CTL_ADD,cne->mfd,&ev);
	}
	if (isReadEv(cne->mwatchEvent))
	{
		mvRCNE[i] = cne;
		atomicInc(cne); //++
		mcneNum++;
	}
	if (isWriteEv(cne->mwatchEvent))
	{
		mvWCNE[i] = cne;
		atomicInc(cne); //++
		mcneNum++;
	}
	mlockCNE.Unlock();
}

void CNetThread::cneStop(cms_net_ev *cne)
{
	bool isRemove = false;
	int i = vectorIdx(cne->mfd);
	mlockCNE.Lock();
// 	printf(">>>>>CNetThread::cneStop %p .\n",this);
	if (mvRCNE[i] != NULL)
	{
		isRemove = true;
		epoll_ctl(mepfd,EPOLL_CTL_DEL,cne->mfd,NULL);
		atomicDec(mvRCNE[i]);
		mvRCNE[i] = NULL;
		mcneNum--;
	}
	if (mvWCNE[i] != NULL)
	{
		if (!isRemove)
		{
			epoll_ctl(mepfd,EPOLL_CTL_DEL,cne->mfd,NULL);
		}
		atomicDec(mvWCNE[i]);
		mvWCNE[i] = NULL;
		mcneNum--;
	}
	mlockCNE.Unlock();
}

int CNetThread::cneSize()
{
	return mcneNum;
}