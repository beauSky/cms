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
#include <net/cms_net_var.h>

void atomicInc(cms_net_ev *cne)
{
	if (cne)
	{
		__sync_add_and_fetch(&cne->monly,1);		//当数据超时，且没人使用时，删除
	}
}

void atomicDec(cms_net_ev *cne)
{
	if (__sync_sub_and_fetch(&cne->monly,1) == 0)//当数据超时，且没人使用时，删除
	{
		delete cne;
	}
}

cms_net_ev *mallcoCmsNetEv()
{
	cms_net_ev * cne = new cms_net_ev;
	cne->monly = 0;
	//cne->mcallBack = NULL;
	cne->mfd = -1;
	cne->mwatchEvent = 0;
	atomicInc(cne); //新创建，计数器加1
	return cne;
}

void freeCmsNetEv(cms_net_ev *cne)
{
	atomicDec(cne);
}

void initCmsNetEv(cms_net_ev *cne,cms_net_cb callback,int fd,int event)
{
	cne->mcallBack = callback;
	cne->mfd = fd;
	cne->mwatchEvent = event;
}

bool isUdpAddrEmpty(UdpAddr ua)
{
	if (ua.miBindPort == 0 && ua.miPort == 0 && ua.muiAddr == 0 && ua.mlistener == NULL)
	{
		return true;
	}
	return false;
}
