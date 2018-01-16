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
#ifndef __CMS_NET_MGR_H__
#define __CMS_NET_MGR_H__
#include <net/cms_net_thread.h>
#include <common/cms_type.h>
#include <core/cms_thread.h>

class CNetMgr
{
public:
	CNetMgr();
	~CNetMgr();

	static CNetMgr *instance();
	static void freeInstance();

	void stop();
// 	static void *routinue(void *param);
// 	void thread();
// 	bool run();
// 	void stop();
	//投递的时间必须包括读或写时间!!!!!!!
	//改变事件之前必须先删除
	void cneStart(cms_net_ev *cne,bool isListen = false);
	void cneStop(cms_net_ev *cne);	
private:
	static CNetMgr *minstance;
	CLock	mlockNetThread;
	std::vector<CNetThread *> mvnetThread;
// 	bool			misRun;
// 	cms_thread_t	mtid;
};
#endif
