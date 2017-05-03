#ifndef __CMS_CONN_H__
#define __CMS_CONN_H__
#include <interface/cms_read_write.h>
#include <interface/cms_interf_conn.h>
#include <core/cms_buffer.h>
#include <protocol/cms_rtmp.h>
#include <common/cms_var.h>
#include <protocol/cms_amf0.h>
#include <flvPool/cms_flv_pool.h>
#include <protocol//cms_flv_transmission.h>
#include <common/cms_type.h>
#include <string>

class CRtmpProtocol;
class CFlvTransmission;
class CConnRtmp:public Conn
{
public:
	CConnRtmp(RtmpType rtmpType,CReaderWriter *rw,std::string pullUrl,std::string pushUrl);
	~CConnRtmp();

	int doit();
	int handleEv(FdEvents *fe);
	int stop(std::string reason);
	std::string getUrl();
	std::string getPushUrl();
	std::string getRemoteIP();
	int doDecode(){return 0;};
	int doTransmission();
	int sendBefore(const char *data,int len){return 0;};

	struct ev_loop  *evLoop();
	struct ev_io    *evReadIO();
	struct ev_io    *evWriteIO();

	void setEVLoop(struct ev_loop *loop);
	
	void setUrl(std::string url);		//拉流或者被推流或者被播放的地址
	void setPushUrl(std::string url);	//推流到其它服务的推流地址
	int  decodeMessage(RtmpMessage *msg);
	int  decodeMetaData(amf0::Amf0Block *block);
	int  decodeSetDataFrame(amf0::Amf0Block *block);
	int  setPublishTask();
	int  setPlayTask();
	void tryCreateTask();
private:
	int  decodeVideo(RtmpMessage *msg,bool &isSave);
	int  decodeAudio(RtmpMessage *msg,bool &isSave);
	int  decodeVideoAudio(RtmpMessage *msg);
	int	 doRead();
	int	 doWrite(bool isTimeout);
	void copy2Slice(Slice *s);
	void makeHash();
	void makePushHash();
	void justTick();

	struct ev_loop	*mloop;			//全局不属于本类
	struct ev_io	*mwatcherReadIO;	//虽然由外面创建 cms_conn_mgr 或者 cms_net_dispatch 但是最终属于本类
	struct ev_io	*mwatcherWriteIO;	//虽然由外面创建 cms_conn_mgr 或者 cms_net_dispatch 但是最终属于本类
	struct ev_timer *mwatcherTimer;

	uint64	mjustTickOld;
	uint64	mjustTick;

	bool		misStop;
	RtmpType	mrtmpType;
	//流信息
	int   miWidth;
	int   miHeight;
	int   miMediaRate;
	int   miVideoFrameRate;
	int   miVideoRate;
	int   miAudioFrameRate;
	int   miAudioRate;
	int   miAudioSamplerate;
	byte  mvideoType;
	byte  maudioType;
	int   miFirstPlaySkipMilSecond;
	bool  misResetStreamTimestamp;	
	bool  misNoTimeout;
	int   miLiveStreamTimeout;
	int   miNoHashTimeout;
	bool  misRealTimeStream;
	int64 mllCacheTT;

	bool misChangeMediaInfo;

	CRtmpProtocol	*mrtmp;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;
	CReaderWriter	*mrw;
	std::string		murl;
	std::string		mremoteAddr;
	std::string		mremoteIP;
	std::string		mHost;
	HASH			mHash;
	uint32          mHashIdx;

	std::string		mstrHash;

	std::string		mstrPushUrl;
	HASH			mpushHash;

	bool			misPublish;
	bool			misPlay;
	bool            misPushFlv;
	bool			misPush;

	int64           mllIdx;

	CFlvTransmission *mflvTrans;
};
#endif
