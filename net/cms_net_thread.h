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
#ifndef __CMS_NET_THREAD_H__
#define __CMS_NET_THREAD_H__
#include <core/cms_thread.h>
#include <common/cms_var.h>
#include <core/cms_lock.h>
#include <net/cms_net_var.h>
#include <vector>
#include <sys/socket.h>
#include <sys/epoll.h>

//每一个 CNetThread 处理 MAX_NET_THREAD_NUM socket
class CNetThread
{
public:
	CNetThread();
	~CNetThread();

	static void *routinue(void *param);
	void thread();
	bool run();
	void stop();

	void cneStart(cms_net_ev *cne,bool isListen = false);
	void cneStop(cms_net_ev *cne);
	int  cneSize();
private:
	bool isReadEv(int evs);
	bool isWriteEv(int evs);
	int  vectorIdx(int fd);
	cms_net_ev *getReadCne(int fd);
	cms_net_ev *getWriteCne(int fd);
	int  epollEV(int evs,bool isListen);

	CLock	mlockCNE;
	std::vector<cms_net_ev *> mvRCNE;
	std::vector<cms_net_ev *> mvWCNE;
	int		mcneNum;
	bool			misRun;
	cms_thread_t	mtid;
	int				mepfd;
};
#endif
