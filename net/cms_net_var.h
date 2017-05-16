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
#ifndef __CMS_NET_VAR_H__
#define __CMS_NET_VAR_H__

#define MAX_NET_THREAD_FD_NUM	1000

#define	EventRead		0x01
#define EventWrite		0x02
#define EventWait2Read  0x04
#define EventWait2Write 0x08
#define EventJustTick   0x10
#define EventErrot		0x400

typedef void(*cms_net_cb)(void *t,int);
typedef struct _cms_net_ev
{
	int			mfd;	
	int			mwatchEvent;
	int			monly;				//0 表示没被使用，大于0表示正在被使用次数
	cms_net_cb	mcallBack;
}cms_net_ev;

void atomicInc(cms_net_ev *cne);
void atomicDec(cms_net_ev *cne);

cms_net_ev *mallcoCmsNetEv();
void	 freeCmsNetEv(cms_net_ev *cne);
void	 initCmsNetEv(cms_net_ev *cne,cms_net_cb callback,int fd,int event);

#endif