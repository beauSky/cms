#include <static/cms_static_common.h>
#include <static/cms_static.h>
#include <common/cms_utility.h>
#include <log/cms_log.h>

OneTask *newOneTask()
{
	OneTask *otk = new OneTask();
	otk->mdownloadTotal = 0;
	otk->mdownloadTick = 0;
	otk->mdownloadSpeed = 0;
	otk->mdownloadTT = getTickCount();

	otk->muploadTotal = 0;
	otk->muploadTick = 0;
	otk->muploadSpeed = 0;
	otk->muploadTT = getTickCount();

	otk->mvideoFramerate = 0;
	otk->maudioFramerate = 0;
	otk->maudioSamplerate = 0;
	otk->mmediaRate = 0;

	otk->mtotalConn = 0;		//该任务当前连接数

	otk->mtotalMem = 0;		//当前任务数据占用内存

	otk->mttCreate = getTimeUnix();
	return otk;
}

void makeOneTaskDownload(HASH &hash,int32 downloadBytes,bool isRemove)
{
	OneTaskDownload *otd = new OneTaskDownload;
	otd->packetID = PACKET_ONE_TASK_DOWNLOAD;
	otd->hash = hash;
	otd->downloadBytes = downloadBytes;
	otd->isRemove = isRemove;

	CStatic::instance()->push((OneTaskPacket *)otd);
}

void makeOneTaskupload(HASH	&hash,int32 uploadBytes,int connAct)
{
	OneTaskUpload *otu = new OneTaskUpload;
	otu->packetID = PACKET_ONE_TASK_UPLOAD;
	otu->hash = hash;
	otu->uploadBytes = uploadBytes;
	otu->connAct = connAct;
	CStatic::instance()->push((OneTaskPacket *)otu);
}

void makeOneTaskMedia(HASH	&hash,int32 videoFramerate,int32 audioFramerate,
					  int32 audioSamplerate,int32 mediaRate,std::string videoType,
					  std::string audioType,std::string url,std::string remoteAddr)
{
	OneTaskMeida *otm = new OneTaskMeida;
	otm->packetID = PACKET_ONE_TASK_MEDA;
	otm->hash = hash;
	otm->videoFramerate = videoFramerate;
	otm->audioFramerate = audioFramerate;
	otm->audioSamplerate = audioSamplerate;
	otm->mediaRate = mediaRate;
	otm->videoType = videoType;
	otm->audioType = audioType;
	otm->remoteAddr = remoteAddr;
	otm->url = url;
	CStatic::instance()->push((OneTaskPacket *)otm);
}

void makeOneTaskMem(HASH	&hash,int64	totalMem)
{
	OneTaskMem *otm = new OneTaskMem;
	otm->packetID = PACKET_ONE_TASK_MEM;
	otm->hash = hash;
	otm->totalMem = totalMem;
	CStatic::instance()->push((OneTaskPacket *)otm);
}
