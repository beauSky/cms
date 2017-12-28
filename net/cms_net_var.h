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
#include <common/cms_utility.h>
#include <string>

#define MAX_NET_THREAD_FD_NUM	1000

#define	EventRead		0x01
#define EventWrite		0x02
#define EventWait2Read  0x04
#define EventWait2Write 0x08
#define EventJustTick   0x10
#define EventErrot		0x400

typedef struct _UdpAddr{
public:
	int miBindPort;
	int miPort;
	unsigned int muiAddr;
	std::string msAddr;
	void *mlistener;
	_UdpAddr()
	{
		miPort = 0;
		muiAddr = 0;
		miBindPort = 0;
		mlistener = NULL;
	}
	_UdpAddr(int iport,unsigned int iaddr,int ibindPort,void *listener)
	{
		miBindPort = ibindPort;
		miPort = iport;
		muiAddr = iaddr;
		mlistener = listener;
		char szIP[30] = {0};
		ipInt2ipStr(iaddr,szIP);
		msAddr = szIP;
	}	
	bool operator < (_UdpAddr const& _A) const
	{
		//这个函数指定排序策略
		if(muiAddr == _A.muiAddr)
		{
			if (miPort == _A.miPort)
			{
				return miBindPort < _A.miBindPort;
			}
			else
			{
				return miPort < _A.miPort;
			}
		}
		else
		{
			return muiAddr < _A.muiAddr;
		}
	}
	bool operator == (_UdpAddr const& _A) const
	{
		if(muiAddr == _A.muiAddr && miPort == _A.miPort && miBindPort == _A.miBindPort)
			return true;
		else
			return false;
	}
	bool operator != (_UdpAddr const& _A) const
	{
		if(muiAddr == _A.muiAddr && miPort == _A.miPort && miBindPort == _A.miBindPort)
			return false;
		else
			return true;
	}
	_UdpAddr& operator = (_UdpAddr const& _A) 
	{
		muiAddr = _A.muiAddr;
		miPort = _A.miPort;
		miBindPort = _A.miBindPort;
		mlistener = _A.mlistener;
		return *this;
	}
} UdpAddr;

bool isUdpAddrEmpty(UdpAddr ua);

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