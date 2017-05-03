#include <dnscache/cms_dns_cache.h>
#include <common/cms_utility.h>
#include <log/cms_log.h>
#include <netdb.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>

#define CdnTimeout (1000*60*5)
CDnsCache* CDnsCache::minstance = NULL;
CDnsCache::CDnsCache()
{

}

CDnsCache::~CDnsCache()
{

}

CDnsCache *CDnsCache::instance()
{
	if (minstance == NULL)
	{
		minstance = new CDnsCache();
	}
	return minstance;
}

void CDnsCache::freeInstance()
{
	if (minstance != NULL)
	{
		delete minstance;
		minstance = NULL;
	}
}

bool CDnsCache::host2ip(const char* host,unsigned long &ip)
{
	if (isLegalIp(host))
	{
		ip = ipStr2ipInt(host);
	}
	else
	{
		unsigned long tt = getTickCount();
		bool needGetHost = true;
		mHostInfoLock.Lock();
		MapHostInfoIter it = mmapHostInfo.find(host);
		if (it != mmapHostInfo.end() && tt - it->second->tt < CdnTimeout)
		{
			ip = it->second->ip;
			needGetHost = false;
		}
		mHostInfoLock.Unlock();
		if (needGetHost)
		{
			struct hostent *hp;
			struct sockaddr_in saddr;
			if ((hp = gethostbyname(host)) == NULL)
			{
				logs->error("*** [CDnsCache::host2ip] can't resolve address: %s,errno=%d,strerrno=%s ***",host,errno,strerror(errno));
				return false;
			}
			memcpy(&saddr.sin_addr, hp->h_addr, hp->h_length);
			ip = saddr.sin_addr.s_addr;
			char szIP[23] = {0};
			ipInt2ipStr(ip,szIP);
			logs->debug(">>>>>>[CDnsCache::host2ip] host %s gethostbyname ip %s <<<<<",host,szIP);
			mHostInfoLock.Lock();
			MapHostInfoIter it = mmapHostInfo.find(host);
			if (it != mmapHostInfo.end())
			{
				it->second->ip = ip;
				it->second->tt = tt;
			}
			else
			{
				HostInfo *hi = new(HostInfo);
				hi->ip = ip;
				hi->tt = tt;
				mmapHostInfo.insert(make_pair(host,hi));
			}
			mHostInfoLock.Unlock();
		}
	}
	return true;
}
