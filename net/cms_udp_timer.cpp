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
#include <net/cms_udp_timer.h>
#include <common/cms_utility.h>
#include <core/cms_lock.h>
#include <common/cms_utility.h>
#include <core/cms_thread.h>
#include <log/cms_log.h>
#include <queue>
#include <assert.h>

#define TIME_UDP_MIL_SECOND	30
queue<cms_udp_timer *> gqueueUDP[TIME_UDP_THREAD_NUM];
CLock gqueueUDPL[TIME_UDP_THREAD_NUM];
bool  gqueueUDPB[TIME_UDP_THREAD_NUM] = { false };
cms_thread_t gqueueUDPT[TIME_UDP_THREAD_NUM] = { 0 };

uint64 gudpUid = 0;
CLock gudpUidLock;

typedef struct  _thread_param
{
	uint64 i;
}thread_param;

void atomicUdpInc(cms_udp_timer *ct)
{
	__sync_add_and_fetch(&ct->only, 1);//当数据超时，且没人使用时，删除
}

void atomicUdpDec(cms_udp_timer *ct)
{
	if (__sync_sub_and_fetch(&ct->only, 1) == 0)//当数据超时，且没人使用时，删除
	{
		delete ct;
	}
}

uint64 getUdpUid()
{
	uint64 uid;
	gudpUidLock.Lock();
	uid = gudpUid++;
	gudpUidLock.Unlock();
	return uid;
}

cms_udp_timer *mallcoCmsUdpTimer()
{
	cms_udp_timer * ct = new cms_udp_timer;
	ct->only = 0;
	atomicUdpInc(ct); //新创建，计数器加1
	return ct;
}

void freeCmsUdpTimer(cms_udp_timer *ct)
{
	atomicUdpDec(ct);
}

void cms_udp_timer_init(cms_udp_timer *ct, int fd, cms_udp_timer_cb cb, uint64 uid)
{
	assert(ct != NULL);
	ct->fd = fd;
	ct->uid = uid;
	ct->cb = cb;
	ct->start = (int64)getTickCount();
	ct->tick = (int64)getTickCount() + TIME_UDP_MIL_SECOND;
}

void cms_timer_udp_thread_stop()
{
	logs->debug("##### cms_timer_udp_thread_stop begin #####");
	for (int i = 0; i <TIME_UDP_THREAD_NUM; i++)
	{
		gqueueUDPB[i] = false;
		cmsWaitForThread(gqueueUDPT[i], NULL);
		gqueueUDPT[i] = 0;
	}
	logs->debug("##### cms_timer_udp_thread_stop finish #####");
}

void *cms_timer_udp_thread(void *param)
{
	thread_param *tp = (thread_param *)param;
	uint64 i = tp->i;
	delete tp;
	logs->info("##### cms_timer_udp_thread enter thread[%d]=%d ###", i, gettid());
	cms_udp_timer *ct;
	bool is;
	int64  mils = 0;
	int64 t;

#ifdef __CMS_APP_DEBUG__
	int runTime = 0;
	long t1 = getTickCount();
	long t2 = t1;
#endif // __CMS_APP_DEBUG__	

	do
	{
		is = false;
		mils = 10;

#ifdef __CMS_APP_DEBUG__
		t2 = getTickCount();
		if (t2 - t1 > 1000)
		{
			t1 = t2;
			logs->info("##### cms_timer_udp_thread runTime=%d ###", runTime);
			runTime = 0;
		}
#endif // __CMS_APP_DEBUG__		

		gqueueUDPL[i].Lock();
		if (!gqueueUDP[i].empty())
		{
			t = (int64)getTickCount();
			ct = gqueueUDP[i].front();
			if (t > ct->tick/*+mils-1*/)
			{
				is = true;
				gqueueUDP[i].pop();
				if (!gqueueUDP[i].empty()) //看需要休眠多长时间
				{
					mils = gqueueUDP[i].front()->tick - t;
				}
			}
		}
		gqueueUDPL[i].Unlock();

		if (is)
		{
			ct->cb(ct);

#ifdef __CMS_APP_DEBUG__
			runTime++;
#endif
		}
		if (mils > 0)
		{
			cmsSleep(mils);
		}
	} while (gqueueUDPB[i]);
	logs->info("##### cms_timer_udp_thread leave thread=[%d]=%d ###", i, gettid());
	return NULL;
}

void cms_udp_timer_start(cms_udp_timer *ct)
{
	int fd = ct->fd;
	if (fd < 0)
	{
		fd = -fd;
	}
	uint64 i = (uint64)(fd % TIME_UDP_THREAD_NUM);
	atomicUdpInc(ct); //投递使用，计数器加1
	ct->tick = (int64)getTickCount() + TIME_UDP_MIL_SECOND;
	gqueueUDPL[i].Lock();
	if (gqueueUDPT[i] == 0)
	{
		gqueueUDPB[i] = true;
		thread_param *tp = new thread_param();
		tp->i = i;
		cmsCreateThread(&gqueueUDPT[i], cms_timer_udp_thread, tp, false);
	}
	if (gqueueUDPB[i])
	{
		gqueueUDP[i].push(ct);
	}
	else
	{
		atomicUdpDec(ct);
	}
	gqueueUDPL[i].Unlock();
}

