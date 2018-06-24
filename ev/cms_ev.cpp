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
#include <ev/cms_ev.h>
#include <common/cms_utility.h>
#include <dispatch/cms_net_dispatch.h>
#include <core/cms_lock.h>
#include <common/cms_utility.h>
#include <log/cms_log.h>
#include <queue>
#include <assert.h>
using namespace std;

#define TIME_OUT_MIL_SECOND	10
#define TIME_IN_MIL_SECOND	10
//写超时
queue<cms_timer *> gqueueWT;
CLock gqueueWL;
bool  gqueueWR = false;
cms_thread_t gqueueWTT = 0;
//读超时
queue<cms_timer *> gqueueRT;
CLock gqueueRL;
bool  gqueueRR = false;
cms_thread_t gqueueRTT = 0;

void atomicInc(cms_timer *ct)
{
	__sync_add_and_fetch(&ct->only,1);//当数据超时，且没人使用时，删除
}

void atomicDec(cms_timer *ct)
{
	if (__sync_sub_and_fetch(&ct->only,1) == 0)//当数据超时，且没人使用时，删除
	{
		delete ct;
	}
}

cms_timer *mallcoCmsTimer()
{
	cms_timer * ct = new cms_timer;
	ct->only = 0;
	atomicInc(ct); //新创建，计数器加1
	return ct;
}

void freeCmsTimer(cms_timer *ct)
{
	atomicDec(ct);
}

void *cms_timer_write_thread(void *param)
{
	logs->info("##### cms_timer_write_thread enter thread=%d ###", gettid());
	cms_timer *ct;
	bool is;
	long long  mils = 0;
	long long t;
	do 
	{
		is = false;
		mils = 5;
		
		gqueueWL.Lock();
		if (!gqueueWT.empty())
		{
			t = (long long)getTickCount();			
			ct = gqueueWT.front();
			if (t > ct->tick/*+mils-1*/)
			{
				is = true;				
				gqueueWT.pop();
				if (!gqueueWT.empty()) //看需要休眠多长时间
				{
					mils = gqueueWT.front()->tick - t;
				}
			}				
		}
		gqueueWL.Unlock();

		if (is)
		{
			ct->cb(ct);
		}
		if (mils > 0)
		{
			cmsSleep(mils);
		}
	} while (gqueueWR);
	logs->info("##### cms_timer_write_thread leave thread=%d ###", gettid());
	return NULL;
}

void cms_ev_timer_thread_stop()
{
	logs->debug("##### cms_ev_timer_thread_stop begin #####");
	gqueueRR = false;
	gqueueWR = false;
	cmsWaitForThread(gqueueRTT, NULL);
	gqueueRTT = 0;
	cmsWaitForThread(gqueueWTT, NULL);
	gqueueWTT = 0;
	logs->debug("##### cms_ev_timer_thread_stop finish #####");
}

void *cms_timer_read_thread(void *param)
{
	logs->info("##### cms_timer_read_thread enter thread=%d ###", gettid());
	cms_timer *ct;
	bool is;
	long long  mils = 0;
	long long t;
	do 
	{
		is = false;
		mils = 10;

		gqueueRL.Lock();
		if (!gqueueRT.empty())
		{
			t = (long long)getTickCount();
			ct = gqueueRT.front();
			if (t > ct->tick)
			{
				is = true;				
				gqueueRT.pop();
				if (!gqueueRT.empty()) //看需要休眠多长时间
				{
					mils = gqueueRT.front()->tick - t;
				}
			}				
		}
		gqueueRL.Unlock();

		if (is)
		{
			ct->cb(ct);
		}
		if (mils > 0)
		{
			cmsSleep(mils);
		}
	} while (gqueueRR);
	logs->info("##### cms_timer_read_thread leave thread=%d ###", gettid());
	return NULL;
}

void cms_timer_init(cms_timer *ct,int fd,cms_timer_cb cb,bool isWrite/* = true*/,int64 uid /*= 0*/)
{
	assert(ct != NULL);
	ct->fd = fd;
	ct->uid = uid;
	ct->cb = cb;
	if (isWrite)
	{
		ct->tick = (long long)getTickCount()+TIME_OUT_MIL_SECOND;
	}
	else
	{
		ct->tick = (long long)getTickCount()+TIME_IN_MIL_SECOND;
	}
}

void cms_timer_start(cms_timer *ct,bool isWrite/* = true*/)
{
	atomicInc(ct); //投递使用，计数器加1
	if (isWrite)
	{
		ct->tick = (long long)getTickCount()+TIME_OUT_MIL_SECOND;
		gqueueWL.Lock();
		if (gqueueWTT == 0)
		{
			gqueueWR = true;
			cmsCreateThread(&gqueueWTT,cms_timer_write_thread,NULL,false);		
		}
		if (gqueueWR)
		{
			gqueueWT.push(ct);
		}
		else
		{
			atomicDec(ct);
		}
		gqueueWL.Unlock();
	}
	else
	{
		ct->tick = (long long)getTickCount()+TIME_IN_MIL_SECOND;
		gqueueRL.Lock();
		if (gqueueRTT == 0)
		{
			gqueueRR = true;
			cmsCreateThread(&gqueueRTT,cms_timer_read_thread,NULL, false);
		}
		if (gqueueRR)
		{
			gqueueRT.push(ct);
		}
		else
		{
			atomicDec(ct);
		}		
		gqueueRL.Unlock();
	}	
}

void acceptEV(void *w,int revents)
{
	cms_net_ev *watcher = (cms_net_ev *)w;
	CNetDispatch::instance()->dispatchAccept(watcher,watcher->mfd);
}

void readEV(void *w,int revents)
{
	cms_net_ev *watcher = (cms_net_ev *)w;
	CNetDispatch::instance()->dispatchEv(watcher,NULL,watcher->mfd,revents);
}

void writeEV(void *w,int revents)
{	
	cms_net_ev *watcher = (cms_net_ev *)w;
	CNetDispatch::instance()->dispatchEv(NULL,watcher,watcher->mfd,revents);
}

void wait2ReadEV(void *t)
{
	cms_timer *ct = (cms_timer *)t;
	CNetDispatch::instance()->dispatchEv(ct,ct->fd,EventWait2Read);
}

void wait2WriteEV(void *t)
{
	cms_timer *ct = (cms_timer *)t;
	CNetDispatch::instance()->dispatchEv(ct,ct->fd,EventWait2Write);
}
