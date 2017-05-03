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
