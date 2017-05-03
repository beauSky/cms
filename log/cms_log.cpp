#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

CLog *cmsLog = NULL;
CLog* cmsLogInstance()
{
	return cmsLog;
}

void cmsLogInit(string dir,LogLevel level,bool console,int limitSize)
{
	cmsLog = new(CLog);
	if (dir.empty())
	{
		dir = "./log/";
	}
	cmsLog->run(dir,level,console,limitSize);
}

int _vscprintf (const char * format, va_list pargs)
{ 
	int retval; 
	va_list argcopy;
	va_copy(argcopy, pargs); 
	retval = vsnprintf(NULL, 0, format, argcopy); 
	va_end(argcopy); 
	return retval;
}

#define MakeLog(fmt,str,level,day,len) \
 {\
	va_list argptr;\
	va_start(argptr, fmt);\
	len = _vscprintf( fmt, argptr)+256;\
	str = new char[len];\
	memset(str,0,len);\
	getLogInfo(level,str);\
	day = getTimeStr(str+strlen(str));\
	vsnprintf(str+strlen(str),len-strlen(str),fmt, argptr);\
	len=strlen(str);}

#define NewLogInfo(logInfo,str,day,len) \
	{\
		logInfo = new(LogInfo);\
		logInfo->day = day;\
		logInfo->len = len;\
		logInfo->log = str;}

void getLogInfo(LogLevel level,char* str)
{
	switch(level)
	{
	case  INFO:
		sprintf(str,"info ");
		break;		
	case  DEBUG:
		sprintf(str,"debug ");
		break;		
	case  FATAL:
		sprintf(str,"fatal ");
		break;		
	case  WARN:
		sprintf(str,"warn ");
		break;
	case ERROR1:
		sprintf(str,"error ");
		break;
	default:
		break;
	}
}

CLog::CLog()
{
	mfp = NULL;
	mfiseSize = 0;
	mlimitSize = 1024*1024*500;
	midx = 0;
}

void *CLog::routinue(void *param)
{
	CLog *pLog = (CLog*)param;
	pLog->thread();
	return NULL;
}

void CLog::thread()
{
	printf(">>>>> CLog thread pid=%ld\n",gettid());
	cmsMkdir(mdir.c_str());
	char szTime[21] = {0};
	getDateTime(szTime);
	if (mdir.at(mdir.length()-1) != '/')
	{
		mdir.append("/");
	}
	mname = getFileName(szTime);
	mfp = fopen((mdir+mname).c_str(),"a+");
	LogInfo* logInfo;
	int day = getTimeDay();
	do 
	{
		logInfo = NULL;
		bool res = pop(&logInfo);
		if (res)
		{
			if (logInfo != NULL)
			{
				if (day != logInfo->day)
				{
					fclose(mfp);
					memset(szTime,0,sizeof(szTime));
					getDateTime(szTime);
					mname = getFileName(szTime);
					mfp = fopen((mdir+mname).c_str(),"a+");
					mfiseSize = 0;
					day = logInfo->day;
				}
				else if (mfiseSize > mlimitSize)
				{
					fclose(mfp);
					mname = getFileName(szTime);
					mfp = fopen((mdir+mname).c_str(),"a+");
					mfiseSize = 0;
				}
				mfiseSize += logInfo->len;
				fwrite(logInfo->log,1,logInfo->len,mfp);
				fwrite("\n",1,1,mfp);
				fflush(mfp);
				if (mconsole)
				{
					printf("%s\n",logInfo->log);
				}				
			}
			if (logInfo->log)
			{
				delete[] logInfo->log;
			}
			delete logInfo;
		}
		else
		{
			cmsSleep(100);
		}
	} while (misRun);
	if (mfp != NULL)
	{
		fclose(mfp);
	}
	printf(">>>>> CLog thread leave pid=%ld\n",gettid());
}

string  CLog::getFileName(char *szDTime)
{
	string fileName;
	char name[128];	
	do 
	{
		snprintf(name,sizeof(name),"%s_%d.log",szDTime,midx);
		fileName = name;
		midx++;
		if (access((mdir+fileName).c_str(), 0) == 0)
		{
			printf(">>>>>log file=%s is exist.\n", fileName.c_str());
			continue;
		}
		break;
	} while (1);
	return fileName;
}

bool CLog::run(string dir,LogLevel level,bool console,int limitSize)
{
	mconsole = console;
	mlevel = level;
	if (limitSize < 1024*1024*1024)
	{
		mlimitSize = limitSize;
	}
	mdir = dir;
	misRun = true;
	int res = cmsCreateThread(&mtid,routinue,this,true);
	if (res == -1)
	{
		char date[128] = {0};
		getTimeStr(date);
		printf("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
		return false;
	}
	return true;
}

void CLog::push(LogInfo* logInfo)
{
	mqueueLock.Lock();
	mqueueLog.push(logInfo);
	mqueueLock.Unlock();
}

bool CLog::pop(LogInfo** logInfo)
{
	bool res = false;
	mqueueLock.Lock();
	if (!mqueueLog.empty())
	{
		*logInfo = mqueueLog.front();
		mqueueLog.pop();
		res = true;
	}
	mqueueLock.Unlock();
	return res;
}

void CLog::debug(const char* fmt,...)
{
	if (mlevel < DEBUG)
	{
		return;
	}
	char *str = NULL;
	int day = 0;
	int len = 0;
	MakeLog(fmt,str,DEBUG,day,len);
	LogInfo *logInfo = NULL;
	NewLogInfo(logInfo,str,day,len);
	push(logInfo);	
}

void CLog::info(const char* fmt,...)
{
	if (mlevel < INFO)
	{
		return;
	}
	char *str = NULL;
	int day = 0;
	int len = 0;
	MakeLog(fmt,str,INFO,day,len);
	LogInfo *logInfo = NULL;
	NewLogInfo(logInfo,str,day,len);
	push(logInfo);
}

void CLog::warn(const char* fmt,...)
{
	if (mlevel < WARN)
	{
		return;
	}
	char *str = NULL;
	int day = 0;
	int len = 0;
	MakeLog(fmt,str,WARN,day,len);
	LogInfo *logInfo = NULL;
	NewLogInfo(logInfo,str,day,len);
	push(logInfo);
}

void CLog::error(const char* fmt,...)
{
	if (mlevel < ERROR1)
	{
		return;
	}
	char *str = NULL;
	int day = 0;
	int len = 0;
	MakeLog(fmt,str,ERROR1,day,len);
	LogInfo *logInfo = NULL;
	NewLogInfo(logInfo,str,day,len);
	push(logInfo);
}

void CLog::fatal(const char* fmt,...)
{
	if (mlevel < FATAL)
	{
		return;
	}
	char *str = NULL;
	int day = 0;
	int len = 0;
	MakeLog(fmt,str,FATAL,day,len);
	LogInfo *logInfo = NULL;
	NewLogInfo(logInfo,str,day,len);
	push(logInfo);
}
