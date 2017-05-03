#include <dispatch/cms_net_dispatch.h>
#include <conn/cms_conn_rtmp.h>
#include <conn/cms_conn_mgr.h>
#include <conn/cms_http_s.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <conn/cms_conn_mgr.h>
#include <ev/cms_ev.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>

#define MaxDispatchNum 3000
#define VDSPtr vector<CDispatch*>*
#define MapTcpListenIter map<int,TCPListener *>::iterator

CNetDispatch *CNetDispatch::minstance = NULL;
CNetDispatch::CNetDispatch()
{
	
}

CNetDispatch::~CNetDispatch()
{
	
}

CNetDispatch *CNetDispatch::instance()
{
	if (minstance == NULL)
	{
		minstance = new CNetDispatch();
	}
	return minstance;
}

void CNetDispatch::freeInstance()
{
	if (minstance != NULL)
	{
		delete minstance;
		minstance = NULL;
	}
}

void CNetDispatch::addOneDispatch(int fd, CDispatch *ds)
{
	printf("addOneDispatch fd=%d\n",fd);
	VDSPtr ptr = NULL;
	mdispatchLock.WLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum ��Ϊһ������
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
	}
	else
	{		
		for (unsigned int n = mfdDispatch.size(); n < (unsigned int)i; n++)
		{			
			mfdDispatch.push_back(NULL);
		}
		ptr = new vector<CDispatch*>;
		mfdDispatch.push_back(ptr);
	}
	if (ptr == NULL)
	{
		//��ǰû�п��ٹ��ռ�
		ptr = new vector<CDispatch*>;
		mfdDispatch.at(i) = ptr;
	}
	i = fd % MaxDispatchNum; //ѡ�ж����е�λ��
	for (unsigned int n = ptr->size(); n < (unsigned int)(i+1); n++)  //����Ԥ����λ��
	{
		ptr->push_back(NULL);
	}
	assert(ptr->at(i) == NULL);
	(*ptr)[i] = ds;
	mdispatchLock.UnWLock();
}

void CNetDispatch::delOneDispatch(int fd)
{
	VDSPtr ptr = NULL;
	mdispatchLock.WLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum ��Ϊһ������
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //ѡ�ж����е�λ��
		if ((ptr->size() > (unsigned int)i))
		{
			//assert(ptr->at(i) != NULL);
			(*ptr)[i] = NULL;
		}
	}	
	mdispatchLock.UnWLock();
}

void CNetDispatch::dispatchEv(struct ev_loop *loop,struct ev_io *watcherRead,struct ev_io *watcherWrite,int fd,int events)
{
	VDSPtr ptr = NULL;
	mdispatchLock.RLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum ��Ϊһ������
	assert(mfdDispatch.size() > (unsigned int)i);
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //ѡ�ж����е�λ��
		if ((ptr->size() > (unsigned int)i) && ptr->at(i))
		{
			//assert(ptr->at(i) != NULL);
			ptr->at(i)->pushEv(fd,events,loop,watcherRead,watcherWrite,NULL);
		}
	}	
	mdispatchLock.UnRLock();
}

void CNetDispatch::dispatchEv(struct ev_loop *loop,struct ev_timer *watcher,int fd,int events)
{
	VDSPtr ptr = NULL;
	mdispatchLock.RLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum ��Ϊһ������
	assert(mfdDispatch.size() > (unsigned int)i);
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //ѡ�ж����е�λ��
		if ((ptr->size() > (unsigned int)i) && ptr->at(i))
		{
			//assert(ptr->at(i) != NULL);
			ptr->at(i)->pushEv(fd,events,loop,NULL,NULL,watcher);
		}
	}	
	mdispatchLock.UnRLock();
}

void CNetDispatch::dispatchEv(cms_timer *ct,int fd,int events)
{
	VDSPtr ptr = NULL;
	mdispatchLock.RLock();
	int i = fd / MaxDispatchNum; //MaxDispatchNum ��Ϊһ������
	assert(mfdDispatch.size() > (unsigned int)i);
	if (mfdDispatch.size() > (unsigned int)i)
	{
		ptr = mfdDispatch.at(i);
		i = fd % MaxDispatchNum; //ѡ�ж����е�λ��
		if ((ptr->size() > (unsigned int)i) && ptr->at(i))
		{
			//assert(ptr->at(i) != NULL);
			ptr->at(i)->pushEv(fd,events,ct);
		}
	}	
	mdispatchLock.UnRLock();
}

void CNetDispatch::addOneListenDispatch(int fd, TCPListener *tls)
{
	mdispatchListenLock.WLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	assert(it == mdispatchListen.end());
	mdispatchListen.insert(make_pair(fd,tls));
	mdispatchListenLock.UnWLock();

	struct ev_io *sockIO = new (ev_io);
	ev_io_init(sockIO, acceptEV, fd, EV_READ);
	ev_io_start(CConnMgrInterface::instance()->loop(), sockIO);
}

void CNetDispatch::delOneListenDispatch(int fd)
{
	mdispatchListenLock.WLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	if (it != mdispatchListen.end())
	{
		it->second->stop();
		delete it->second;
		mdispatchListen.erase(it);
	}
	mdispatchListenLock.UnWLock();
}

bool CNetDispatch::nonblocking(int fd)
{
	int opts;
	opts = fcntl(fd, F_GETFL);
	if (opts < 0)
	{
		logs->error("***** sock[ %d ] fcntl(sock,GETFL) *****\n",fd);
		return false;
	}
	opts = opts | O_NONBLOCK;
	if (fcntl(fd, F_SETFL, opts) < 0)
	{
		logs->error("***** sock[ %d ] fcntl(sock,SETFL,opts) *****\n",fd);
		return false;
	}
	return true;
}

void CNetDispatch::dispatchAccept(struct ev_loop *loop,struct ev_io *watcher,int fd)
{
	logs->debug("#### dispatchAccept enter ###");
	int cfd = CMS_ERROR;
	ConnType listenType = TypeNetNone;
	mdispatchListenLock.RLock();
	MapTcpListenIter it = mdispatchListen.find(fd);
	if (it != mdispatchListen.end())
	{
		cfd = it->second->accept();	
		listenType = it->second->listenType();
	}
	mdispatchListenLock.UnRLock();
	if (cfd != CMS_ERROR)
	{
		nonblocking(cfd);
		TCPConn *tcp = new TCPConn(cfd);
		if (listenType == TypeHttp)
		{
			CHttpServer *hs = new CHttpServer(tcp,false);
			if (hs->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,hs);

				struct ev_io *sockIO = new (ev_io);
				ev_io_init(sockIO, readEV, cfd, EV_READ);
				ev_io_start(CConnMgrInterface::instance()->loop(), sockIO);
			}
			else
			{
				delete hs;
			}
		}
		else if (listenType == TypeHttps)
		{
			CHttpServer *hs = new CHttpServer(tcp,true);
			if (hs->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,hs);

				struct ev_io *sockIO = new (ev_io);
				ev_io_init(sockIO, readEV, cfd, EV_READ);
				ev_io_start(CConnMgrInterface::instance()->loop(), sockIO);
			}
			else
			{
				delete hs;
			}
		}
		else if (listenType == TypeRtmp)
		{
			CConnRtmp *rtmp = new CConnRtmp(RtmpServerBPlayOrPublish,tcp,"","");
			if (rtmp->doit() != CMS_ERROR)
			{
				CConnMgrInterface::instance()->addOneConn(cfd,rtmp);

				struct ev_io *sockIO = new (ev_io);
				ev_io_init(sockIO, readEV, cfd, EV_READ);
				ev_io_start(CConnMgrInterface::instance()->loop(), sockIO);
			}
			else
			{
				delete rtmp;
			}
		}
	}
	logs->debug("#### dispatchAccept leave ###");
}


