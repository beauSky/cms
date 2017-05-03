#ifndef __CMS_LOG_H__
#define __CMS_LOG_H__
#include <core/cms_thread.h>
#include <core/cms_lock.h>
#include <stdio.h>
#include <string>
#include <queue>

#define logs cmsLogInstance()
using namespace std;

enum LogLevel 
{
	OFF,
	FATAL,
	ERROR1,
	WARN,
	INFO,
	DEBUG,
	ALL_LEVEL
};

struct LogInfo 
{
	int		day;
	char*	log;
	int     len;
};

class CLog 
{
private:
	cms_thread_t mtid;
	bool		mconsole;
	bool		misRun;
	FILE*		mfp;
	int			mfiseSize;
	int			mlimitSize;
	int			midx;
	LogLevel	mlevel;
	string		mdir;
	string		mname;
	queue<LogInfo *> mqueueLog;
	CLock		mqueueLock;	
	void push(LogInfo* logInfo);
	bool pop(LogInfo** logInfo);
public:
	CLog();
	static void *routinue(void *param);
	string getFileName(char *szDTime);
	void thread();
	bool run(string dir,LogLevel level,bool console,int limitSize = 1024*1024*500);
	void debug(const char* fmt,...);
	void info(const char* fmt,...);
	void warn(const char* fmt,...);
	void error(const char* fmt,...);
	void fatal(const char* fmt,...);		
};

CLog*	cmsLogInstance();
void	cmsLogInit(string dir,LogLevel level,bool console,int limitSize = 1024*1024*500);

#endif