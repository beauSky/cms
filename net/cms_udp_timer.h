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
#ifndef __CMS_UDP_TIMER_H__
#define __CMS_UDP_TIMER_H__
#include <common/cms_type.h>

#define TIME_UDP_THREAD_NUM 8
typedef void(*cms_udp_timer_cb)(void *t);
typedef struct _cms_udp_timer
{
	int    fd;
	uint64 uid;
	int64  tick;
	int	   only;				//0 表示没被使用，大于0表示正在被使用次数
	int64  start;
	cms_udp_timer_cb cb;
}cms_udp_timer;

uint64 getUdpUid();

void atomicUdpInc(cms_udp_timer *ct);
void atomicUdpDec(cms_udp_timer *ct);

cms_udp_timer *mallcoCmsUdpTimer();
void freeCmsUdpTimer(cms_udp_timer *ct);

void cms_udp_timer_init(cms_udp_timer *ct, int fd, cms_udp_timer_cb cb, uint64 uid);
void cms_udp_timer_start(cms_udp_timer *ct);

void cms_timer_udp_thread_stop();

#endif
