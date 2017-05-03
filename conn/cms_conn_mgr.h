#ifndef __CMS_CONN_MGR_H__
#define __CMS_CONN_MGR_H__
#include <interface/cms_dispatch.h>
#include <conn/cms_conn_rtmp.h>
#include <core/cms_lock.h>
#include <core/cms_thread.h>
#include <net/cms_tcp_conn.h>
#include <common/cms_var.h>
#include <map>
#include <queue>
using namespace std;

#define NUM_OF_THE_CONN_MGR 8

class CConnMgr:CDispatch
{
public:
	bool run();
	void stop();
	static void *routinue(void *param);	
	void addOneConn(int fd,Conn *c);
	void delOneConn(int fd);
	void  pushEv(int fd,int events,struct ev_loop *loop,struct ev_io *watcherRead,struct ev_io *watcherWrite,struct ev_timer *watcherTimeout);
	void  pushEv(int fd,int events,cms_timer *ct);
private:	
	void dispatchEv(FdEvents *fe);
	bool popEv(FdEvents **fe);
	void thread();
	map<int,Conn *> mfdConn;
	CRWlock mfdConnLock;

	bool misRun;
	cms_thread_t mtid;
	queue<FdEvents *> mqueue;
	CLock mqueueLock;
};

class CConnMgrInterface
{
public:
	CConnMgrInterface();
	~CConnMgrInterface();

	static CConnMgrInterface *instance();
	static void freeInstance();

	static void *routinue(void *param);
	void thread();
	bool run();

	void addOneConn(int fd,Conn *c);
	void delOneConn(int fd);
	struct ev_loop *loop();
	Conn *createConn(char *addr,string pullUrl,std::string pushUrl,ConnType connectType,RtmpType rtmpType);
private:
	static CConnMgrInterface *minstance;
	CConnMgr *mconnMgrArray[NUM_OF_THE_CONN_MGR];
	struct ev_loop *mloop;
	struct ev_timer*mtimer; 
	bool			misRun;
	cms_thread_t	mtid;
};
#endif
