#include <net/cms_udp_timer.h>
#include <common/cms_utility.h>
#include <core/cms_lock.h>
#include <common/cms_utility.h>
#include <core/cms_thread.h>
#include <log/cms_log.h>
#include <queue>
#include <assert.h>

#define TIME_UDP_MIL_SECOND	10
queue<cms_udp_timer *> gqueueUDP[TIME_UDP_THREAD_NUM];
CLock gqueueUDPL[TIME_UDP_THREAD_NUM];
bool  gqueueUDPB[TIME_UDP_THREAD_NUM] = { false };
cms_thread_t gqueueUDPT[TIME_UDP_THREAD_NUM];

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
	ct->tick = (int64)getTickCount() + TIME_UDP_MIL_SECOND;
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
	do
	{
		is = false;
		mils = 10;

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
	if (!gqueueUDPB[i])
	{
		gqueueUDPB[i] = true;
		thread_param *tp = new thread_param();
		tp->i = i;
		cmsCreateThread(&gqueueUDPT[i], cms_timer_udp_thread, tp, true);
	}
	gqueueUDP[i].push(ct);
	gqueueUDPL[i].Unlock();
}

