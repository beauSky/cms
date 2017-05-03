#ifndef __CMS_EV_H__
#define __CMS_EV_H__
#include <libev/ev.h>
#include <common/cms_var.h>

enum EventType
{	
	EventRead    = 0x01,
	EventWrite   = 0x02,
	EventWait2Read  = 0x04,
	EventWait2Write = 0x08,
	EventJustTick = 0x10,
	EventErrot   = 0x400
	
};

void atomicInc(cms_timer *ct);
void atomicDec(cms_timer *ct);

cms_timer *mallcoCmsTimer();
void	   freeCmsTimer(cms_timer *ct);

void cms_timer_init(cms_timer *ct,int fd,cms_timer_cb cb);
void cms_timer_start(cms_timer *ct);
void wait2ReadEV(void *t);
void wait2WriteEV(void *t);
void justTickEV(struct ev_loop *loop,struct ev_timer *watcher,int revents);

void acceptEV(struct ev_loop *loop,struct ev_io *watcher,int revents);
void readEV(struct ev_loop *loop,struct ev_io *watcher,int revents);
void writeEV(struct ev_loop *loop,struct ev_io *watcher,int revents);
void timerTick(struct ev_loop *loop,struct ev_timer *watcher,int revents);

#endif


