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
#ifndef __CMS_COMMON_TYPE_H__
#define __CMS_COMMON_TYPE_H__
#include <net/cms_net_var.h>

typedef void (*cms_timer_cb)(void *t);
typedef struct _cms_timer 
{ 
	int  fd;
	long long uid;
	long long tick;
	int	 only;				//0 表示没被使用，大于0表示正在被使用次数
	cms_timer_cb cb;
}cms_timer;

struct FdEvents 
{
	int fd;
	int events;
	cms_net_ev *watcherReadIO;
	cms_net_ev *watcherWriteIO;
	struct _cms_timer *watcherWCmsTimer;
	struct _cms_timer *watcherRCmsTimer;
};
#endif