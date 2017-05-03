#ifndef __CMS_TCP_CONN_H__
#define __CMS_TCP_CONN_H__
#include <interface/cms_read_write.h>
#include <core/cms_lock.h>
#include <netinet/in.h>
#include <string>
#include <queue>
using namespace std;

enum ConnType
{
	TypeNetNone,
	TypeHttp,
	TypeHttps,
	TypeRtmp,
	TypeQuery
};

class TCPConn:public CReaderWriter
{
public:
	TCPConn(int fd);
	TCPConn();
	~TCPConn();

	int   dialTcp(char *addr,ConnType connectType);
	int	  connect();
	int   read(char* dstBuf,int len,int &nread);
	int   write(char *srcBuf,int len,int &nwrite);
	void  close();
	int   fd();
	int   remoteAddr(char *addr,int len);
	void  setReadTimeout(long long readTimeout);
	long long getReadTimeout();
	void  setWriteTimeout(long long writeTimeout);
	long long getWriteTimeout();
	long long getReadBytes();
	long long getWriteBytes();	
	ConnType connectType();
	char  *errnoCode();
	int   errnos();
	int   setNodelay(int on);
private:
	int mfd;
	struct sockaddr_in mto;
	string maddr;
	long long mreadTimeout;
	long long mreadBytes;
	long long mwriteTimetou;
	long long mwriteBytes;
	ConnType mconnectType;
	int  merrcode;
};

class TCPListener
{
public:
	TCPListener();
	int  listen(char* addr,ConnType listenType);
	void stop();
	int  accept();
	int  fd();
	ConnType listenType();
private:
	string  mlistenAddr;
	bool	mruning;
	int     mfd;
	ConnType mlistenType;
};
#endif
