#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <log/cms_log.h>

class CAddr
{
public:
	CAddr(char *addr,int defaultPort);
	~CAddr();
	char *addr();
	char *host();
	int  port();
private:
	char	*mAddr;
	char    *mHost;
	int		mPort;
};

class Clog
{
public:
	Clog(char *path,char *level,bool console,int size);
	~Clog();
	char		*path();
	int			size();
	LogLevel	level();
	bool		console();
private:
	char		*mpath;
	int			msize;
	bool		mconsole;
	LogLevel	mlevel;
};

class CConfig
{
public:
	CConfig();
	~CConfig();
	static CConfig *instance();
	static void freeInstance();
	bool	init(const char *configPath);
	CAddr   *addrHttp();
	CAddr	*addrHttps();
	CAddr	*addrRtmp();
	CAddr	*addrQuery();
	Clog	*clog();
private:
	static CConfig *minstance;

	CAddr	*mHttp;
	CAddr	*mHttps;
	CAddr	*mRtmp;
	CAddr	*mQuery;

	Clog	*mlog;
};
#endif
