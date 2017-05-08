#ifndef __CMS_STATIC_H__
#define __CMS_STATIC_H__
#include <common/cms_type.h>
#include <core/cms_lock.h>
#include <core/cms_thread.h>
#include <json/json.h>
#include <static/cms_static_common.h>
#include <string>
#include <map>
#include <queue>

class CStatic
{
public:
	CStatic();
	~CStatic();

	static CStatic *instance();
	static void freeInstance();

	bool run();
	void stop();
	void thread();
	static void *routinue(void *param);	
	void setAppName(std::string appName);

	void push(OneTaskPacket *otp);
	std::string dump();
private:
	bool pop(OneTaskPacket **otp);

	void handle(OneTaskDownload *otd);
	void handle(OneTaskUpload *otu);
	void handle(OneTaskMeida *otm);
	void handle(OneTaskMem *otm);

	int getTaskInfo(Json::Value &value);

	float		getMemUsage();
	int			getMemSize();
	int			getCpuUsage();
	std::string getUploadBytes();
	std::string getDownloadBytes();
	CpuInfo		getCpuInfo();

	static CStatic				*minstance;
	std::map<HASH,OneTask *>	mmapHashTask;
	CLock						mlockHashTask;

	std::queue<OneTaskPacket*>	mqueueOneTaskPacket;
	CLock						mlockOneTaskPacket;

	CLock						mlockDownload;
	int64						mdownloadTick;
	int64						mdownloadSpeed;
	uint64						mdownloadTT;
	CLock						mlockUpload;
	int64						muploadTick;
	int64						muploadSpeed;
	uint64						muploadTT;

	time_t						mappStartTime;
	time_t						mupdateTime;
	int32						mtotalConn;

	bool			misRun;
	cms_thread_t	mtid;

	CpuInfo mcpuInfo0;
	std::string mappName;
};
#endif