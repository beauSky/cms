#include <ev/cms_ev.h>
#include <common/cms_utility.h>
#include <dispatch/cms_net_dispatch.h>
#include <core/cms_lock.h>
#include <common/cms_utility.h>
#include <core/cms_thread.h>
#include <log/cms_log.h>
#include <queue>
#include <assert.h>
using namespace std;

#define TIME_OUT_MIL_SECOND	10
queue<cms_timer *> gqueueT;
CLock gqueueL;
bool  gqueueR = false;
cms_thread_t gqueueTT;

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
	cms_timer * ct = new(cms_timer);
	ct->only = 0;
	atomicInc(ct); //新创建，计数器加1
	return ct;
}

void freeCmsTimer(cms_timer *ct)
{
	atomicDec(ct);
}

int event2event(int revents)
{
	int event = 0;
	if (EV_ERROR&revents)
	{
		event |= EventErrot;
	}
	if (EV_READ&revents)
	{
		event |= EventRead;
	}
	if (EV_WRITE&revents)
	{
		event |= EventWrite;
	}
	return event;
}

void *cms_timer_thread(void *param)
{
	logs->info("##### cms_timer_thread enter #####");
	cms_timer *ct;
	bool is;
	long long  mils = 0;
	long long t;
	do 
	{
		is = false;
		mils = 1;
		
		gqueueL.Lock();
		if (!gqueueT.empty())
		{
			t = (long long)getTickCount();			
			ct = gqueueT.front();
			if (t > ct->tick+mils-1)
			{
				is = true;				
				gqueueT.pop();
				if (!gqueueT.empty()) //看需要休眠多长时间
				{
					mils = gqueueT.front()->tick - t;
				}
			}				
		}
		gqueueL.Unlock();

		if (is)
		{
			ct->cb(ct);
		}
		if (mils > 0)
		{
			cmsSleep(mils);
		}
	} while (gqueueR);
	logs->info("##### cms_timer_thread leave #####");
	return NULL;
}

void cms_timer_init(cms_timer *ct,int fd,cms_timer_cb cb)
{
	assert(ct != NULL);
	ct->fd = fd;
	ct->cb = cb;
	ct->tick = (long long)getTickCount()+TIME_OUT_MIL_SECOND;
}

void cms_timer_start(cms_timer *ct)
{
	ct->tick = (long long)getTickCount()+TIME_OUT_MIL_SECOND;
	atomicInc(ct); //投递使用，计数器加1
	gqueueL.Lock();
	if (!gqueueR)
	{
		gqueueR = true;
		cmsCreateThread(&gqueueTT,cms_timer_thread,NULL,true);		
	}
	gqueueT.push(ct);
	gqueueL.Unlock();
}

void acceptEV(struct ev_loop *loop,struct ev_io *watcher,int revents)
{
	CNetDispatch::instance()->dispatchAccept(loop,watcher,watcher->fd);
}

void readEV(struct ev_loop *loop,struct ev_io *watcher,int revents)
{
	CNetDispatch::instance()->dispatchEv(loop,watcher,NULL,watcher->fd,event2event(revents));
}

void writeEV(struct ev_loop *loop,struct ev_io *watcher,int revents)
{
	CNetDispatch::instance()->dispatchEv(loop,NULL,watcher,watcher->fd,event2event(revents));
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

void justTickEV(struct ev_loop *loop,struct ev_timer *watcher,int revents)
{
	int fd = (long)watcher->data;
	CNetDispatch::instance()->dispatchEv(loop,watcher,fd,EventJustTick);
}

void timerTick(struct ev_loop *loop,struct ev_timer *watcher,int revents)
{
	printf(">>>>>timer tick.\n");
}
