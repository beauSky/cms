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
#ifndef __CMS_EV_H__
#define __CMS_EV_H__
#include <common/cms_var.h>
#include <common/cms_type.h>
#include <net/cms_net_var.h>

void atomicInc(cms_timer *ct);
void atomicDec(cms_timer *ct);

cms_timer *mallcoCmsTimer();
void	   freeCmsTimer(cms_timer *ct);

void cms_timer_init(cms_timer *ct,int fd,cms_timer_cb cb,bool isWrite = true,int64 uid = 0);
void cms_timer_start(cms_timer *ct,bool isWrite = true);
void wait2ReadEV(void *t);
void wait2WriteEV(void *t);

void acceptEV(void *w,int revents);
void readEV(void *w,int revents);
void writeEV(void *w,int revents);

#endif


