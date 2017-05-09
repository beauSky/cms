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
#ifndef __CMS_NET_DISPATCH_H__
#define __CMS_NET_DISPATCH_H__
#include <interface/cms_dispatch.h>
#include <net/cms_tcp_conn.h>
#include <core/cms_lock.h>
#include <ev/cms_ev.h>
#include <libev/ev.h>
#include <vector>
#include <map>
using namespace std;

class CNetDispatch
{
public:
	CNetDispatch();
	~CNetDispatch();
	static CNetDispatch *instance();
	static void freeInstance();

	void addOneDispatch(int fd, CDispatch *ds);
	void delOneDispatch(int fd);
	void dispatchEv(struct ev_loop *loop,struct ev_io *watcherRead,struct ev_io *watcherWrite,int fd,int events);
	void dispatchEv(struct ev_loop *loop,struct ev_timer *watcher,int fd,int events);
	void dispatchEv(cms_timer *ct,int fd,int events);

	void addOneListenDispatch(int fd, TCPListener *tls);
	void delOneListenDispatch(int fd);
	void dispatchAccept(struct ev_loop *loop,struct ev_io *watcher,int fd);
private:
	bool nonblocking(int fd);

	static CNetDispatch *minstance;
	vector<vector<CDispatch*>* > mfdDispatch;
	CRWlock mdispatchLock;

	map<int,TCPListener *> mdispatchListen;
	CRWlock mdispatchListenLock;	
};
#endif
