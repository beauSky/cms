#include <config/cms_config.h>
#include <log/cms_log.h>
#include <conn/cms_conn_mgr.h>
#include <dispatch/cms_net_dispatch.h>
#include <dnscache/cms_dns_cache.h>
#include <common/cms_shmmgr.h>
#include <app/cms_server.h>
#include <flvPool/cms_flv_pool.h>
#include <taskmgr/cms_task_mgr.h>
#include <map>
#include <string>
#include <signal.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/resource.h>
using namespace std;

map<string,string> mapAgrv;
#define ParamConfig "-c"

void sample(char *app)
{
	printf("##### useage: %s -c config.json\n", app);
}

void inorgSignal()
{
	signal(SIGPIPE, SIG_IGN);;
}

int daemon()
{
	int fd;
	int pid;

	switch(fork()) 
	{
	case -1:
		logs->error("%s(%d)-%s: fork() error: %s",
			__FILE__, __LINE__, __FUNCTION__, strerror(errno));
		return -1;

	case 0:
		break;

	default:
		logs->debug("Daemon exit\n");
		exit(0);
	}

	pid = getpid();

	if (setsid() == -1) {
		logs->error("%s(%d)-%s: setsid() failed",
			__FILE__, __LINE__, __FUNCTION__);
		return -1;
	}

	umask(0);

	fd = open("/dev/null", O_RDWR);
	if (fd == -1) {
		logs->error("%s(%d)-%s: open(\"/dev/null\") failed",
			__FILE__, __LINE__, __FUNCTION__);
		return -1;
	}

	if (dup2(fd, STDIN_FILENO) == -1) {
		logs->error( "%s(%d)-%s: dup2(STDIN) failed",
			__FILE__, __LINE__, __FUNCTION__);
		return -1;
	}

	if (dup2(fd, STDOUT_FILENO) == -1) {
		logs->error("%s(%d)-%s: dup2(STDOUT) failed",
			__FILE__, __LINE__, __FUNCTION__);
		return -1;
	}

	if (fd > STDERR_FILENO) {
		if (close(fd) == -1) {
			logs->error( "%s(%d)-%s: close() failed",
				__FILE__, __LINE__, __FUNCTION__);
			return -1;
		}
	}
	return 0;
}

void initInstance()
{
	CConfig::instance();
	CFlvPool::instance();
	CConnMgrInterface::instance();
	CNetDispatch::instance();
	CDnsCache::instance();
	CShmMgr::instance();
	CServer::instance();
	CTaskMgr::instance();
}

void parseVar(int num,char **argv)
{
	for (int i = 0; i < num/2; i++)
	{
		mapAgrv.insert(make_pair(argv[i],argv[i+1]));
	}
}

string getVar(string key)
{
	map<string,string>::iterator it = mapAgrv.find(key.c_str());
	if (it != mapAgrv.end())
	{
		return it->second;
	}
	return "";
}

void cycleServer()
{
	do 
	{
		cmsSleep(1000);
	} while (1);
}

void setRlimit()
{
	struct rlimit rlim;
	rlim.rlim_cur = 300000;
	rlim.rlim_max = 300000;
	setrlimit(RLIMIT_NOFILE, &rlim);

	rlim.rlim_cur = 0;
	rlim.rlim_max = 0;
	getrlimit(RLIMIT_NOFILE, &rlim);
	logs->debug("+++ open file cur %d,open file max %d +++\n",rlim.rlim_cur,rlim.rlim_max);
}

int main(int argc,char *argv[])
{	
	inorgSignal();	
	if (argc % 2 != 1)
	{
		printf("***** main argc is error,should argc %% 2 == 1*****\n");
		return 0;
	}
	parseVar(argc-1,argv+1);
	string config = getVar(ParamConfig);
	if (config.empty())
	{
		sample(argv[0]);
		return 0;
	}
	if (!CConfig::instance()->init(config.c_str()))
	{
		return 0;
	}
	cmsLogInit(CConfig::instance()->clog()->path(),CConfig::instance()->clog()->level(),
		CConfig::instance()->clog()->console(),CConfig::instance()->clog()->size());
	setRlimit();
	//必须先初始化日志，否则会崩溃
	initInstance();	
	if (!CFlvPool::instance()->run())
	{
		logs->error("*** CFlvPool::instance()->run() fail ***");
		cmsSleep(1000*3);
		return 0;
	}
	if (!CConnMgrInterface::instance()->run())
	{
		logs->error("*** CConnMgrInterface::instance()->run() fail ***");
		cmsSleep(1000*3);
		return 0;
	}
	if (!CTaskMgr::instance()->run())
	{
		logs->error("*** CTaskMgr::instance()->run() fail ***");
		cmsSleep(1000*3);
		return 0;
	}
	if (!CServer::instance()->listenAll())
	{
		logs->error("*** CServer::instance()->listenAll() fail ***");
		cmsSleep(1000*3);
		return 0;
	}
	cycleServer();
	logs->debug("cms app exit.");
}