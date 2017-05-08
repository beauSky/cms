/*
��http֧�ּ򵥵�http�����ձ��http-flv����֧��gzip
*/
#ifndef __CMS_HTTP_C_H__
#define __CMS_HTTP_C_H__
#include <interface/cms_interf_conn.h>
#include <interface/cms_read_write.h>
#include <common/cms_type.h>
#include <protocol/cms_http.h>
#include <string>

class ChttpClient:public Conn
{
public:
	ChttpClient(CReaderWriter *rw,std::string pullUrl,std::string oriUrl,
		std::string refer,bool isTls);
	~ChttpClient();

	int doit();
	int handleEv(FdEvents *fe);
	int stop(std::string reason);
	std::string getUrl();
	std::string getPushUrl();
	std::string getRemoteIP();
	struct ev_loop  *evLoop();
	struct ev_io    *evReadIO();
	struct ev_io    *evWriteIO();
	void down8upBytes();

	void setEVLoop(struct ev_loop *loop);

	int doDecode();
	int doReadData();
	int doTransmission();
	int sendBefore(const char *data,int len);

	int doRead();
	int doWrite(bool isTimeout);

private:
	int  request();
	int  handle();
	int	 handleFlv(int &ret);
	void makeHash();
	void copy2Slice(Slice *s);
	void tryCreateTask();
	int  decodeMetaData(char *data,int len);
	int  decodeVideo(char *data,int len,uint32 timestamp);
	int  decodeAudio(char *data,int len,uint32 timestamp);

	struct ev_loop	*mloop;			//ȫ�ֲ����ڱ���
	struct ev_io	*mwatcherReadIO;	
	struct ev_io	*mwatcherWriteIO;

	bool			misRequet;
	bool			misDecodeHeader;
	bool			misRedirect;
	CReaderWriter	*mrw;
	std::string		murl;
	std::string		moriUrl;
	std::string		mredirectUrl;
	std::string		mremoteAddr;
	std::string		mremoteIP;
	std::string		mHost;
	HASH			mHash;
	uint32          mHashIdx;
	std::string		mstrHash;
	std::string		mstrRefer;

	int64           mllIdx;
	CHttp			*mhttp;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;

	//����Ϣ
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

	bool  misChangeMediaInfo;
	bool  misPushFlv;
	bool  misStop;

	FlvPoolDataType miTagType;			//��������
	uint32			muiTimestamp;	    //��slice���ݶ�Ӧrtmp��ʱ���
	int	  miReadFlvHeader;
	bool  misReadTagHeader;
	bool  misReadTagBody;
	bool  misReadTagFooler;
	char  *mtagFlv;
	int	  mtagLen;
	int   mtagReadLen;

	unsigned long  mspeedTick;
};

#endif
