#ifndef __CMS_UDP_TIMER_H__
#define __CMS_UDP_TIMER_H__
#include <common/cms_type.h>

#define TIME_UDP_THREAD_NUM 8
typedef void(*cms_udp_timer_cb)(void *t);
typedef struct _cms_udp_timer
{
	int  fd;
	uint64 uid;
	int64 tick;
	int	 only;				//0 表示没被使用，大于0表示正在被使用次数
	cms_udp_timer_cb cb;
}cms_udp_timer;

uint64 getUdpUid();

void atomicUdpInc(cms_udp_timer *ct);
void atomicUdpDec(cms_udp_timer *ct);

cms_udp_timer *mallcoCmsUdpTimer();
void freeCmsUdpTimer(cms_udp_timer *ct);

void cms_udp_timer_init(cms_udp_timer *ct, int fd, cms_udp_timer_cb cb, uint64 uid);
void cms_udp_timer_start(cms_udp_timer *ct);

#endif
