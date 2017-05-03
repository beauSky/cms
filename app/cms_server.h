#ifndef __APP_SERVER_H__
#define __APP_SERVER_H__
#include <net/cms_tcp_conn.h>

class CServer
{
public:
	CServer();
	~CServer();
	static CServer *instance();
	static void freeInstance();
	bool	listenAll();
private:
	static CServer	*minstance;

	TCPListener	*mHttp;
	TCPListener *mHttps;
	TCPListener	*mRtmp;
	TCPListener	*mQuery;
};
#endif
