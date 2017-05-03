#ifndef __CMS_DNS_CACHE_H__
#define __CMS_DNS_CACHE_H__
#include <core/cms_lock.h>
#include <string>
#include <map>
using namespace std;

struct HostInfo 
{
	unsigned long ip;
	unsigned long tt;
};
#define MapHostInfoIter map<string,HostInfo*>::iterator

class CDnsCache
{
public:
	CDnsCache();
	~CDnsCache();
	static CDnsCache *instance();
	static void freeInstance();
	bool host2ip(const char* host,unsigned long &ip);
private:
	static CDnsCache *minstance;
	map<string,HostInfo*> mmapHostInfo;
	CLock mHostInfoLock;
};
#endif
