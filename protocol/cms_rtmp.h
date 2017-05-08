#ifndef __CMS_RTMP_H__
#define __CMS_RTMP_H__
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <protocol/cms_rtmp_const.h>
#include <core/cms_buffer.h>
#include <protocol/cms_rtmp_handshake.h>
#include <interface/cms_protocol.h>
#include <common/cms_url.h>
#include <common/cms_var.h>
#include <protocol/cms_amf0.h>
#include <conn/cms_conn_rtmp.h>
#include <flvPool/cms_flv_pool.h>
#include <libev/ev.h>
#include <map>
#include <string>

class CConnRtmp;
class CRtmpProtocol:public CProtocol
{
public:
	CRtmpProtocol(void *super,RtmpType rtmpType,CBufferReader *rd,
		CBufferWriter *wr,CReaderWriter *rw,std::string remoteAddr);
	~CRtmpProtocol();

	int want2Read();
	int want2Write(bool isTimeout);
	int wait2Read();
	int wait2Write();
	std::string getRtmpType();
	int decodeChunkSize(RtmpMessage *msg);
	int decodeWindowSize(RtmpMessage *msg);
	int decodeBandWidth(RtmpMessage *msg);
	int handleUserControlMsg(RtmpMessage *msg);
	int decodeAmf03(RtmpMessage *msg,bool isAmf3);
	//�����麯��
	int sendMetaData(Slice *s);
	int sendVideoOrAudio(Slice *s,uint32 uiTimestamp);
	std::string remoteAddr();
	std::string getUrl();
	int writeBuffSize();
	void syncIO();

	void shouldCloseNodelay();
	cms_timer *cmsTimer2Write();
private:
	//����
	int handShake();
	int c2sComplexShakeC0C1();
	int c2sComplexShakeS0();
	int c2sComplexShakeS1C2();
	int c2sComplexShakeS2();
	int s2cSampleShakeC012();
	int s2cComplexShakeC012();	
	//������Ϣ
	int  readMessage();
	int  readBasicHeader(char &fmt,int &cid,int &handleLen);
	int  readRtmpHeader(RtmpHeader &header,int fmt,int &handleLen);
	int  readRtmpPlayload(RtmpHeader &header,int fmt,int cid,int &handleLen);
	int  decodeMessage(RtmpMessage *msg);
	void copyRtmpHeader(RtmpHeader *pDest,RtmpHeader *pSrc);
	int  decodeConnect(amf0::Amf0Block *block);
	int  decodeReleaseStream(amf0::Amf0Block *block);
	int  decodeFcPublish(amf0::Amf0Block *block);
	int  decodePublish(amf0::Amf0Block *block);
	int  decodePlay(amf0::Amf0Block *block);
	int  decodeCreateStream(amf0::Amf0Block *block);
	int  decodeUnPublish(amf0::Amf0Block *block);
	//������Ӧ��rtmp��Ϣ
	int  decodeCommandResult(RtmpCommand cmd);
	//����rtmp��Ϣ
	int doConnect();
	int doReleaseStream();
	int doFCPublish();
	int doCreateStream();
	int doPlay();
	int doPublish();
	int doSetBufLength();
	int doCheckBW();
	int doWindowSize();
	int doBandWidth();
	int doConnectSucc(int event,int objectEncoding);
	int doOnBWDone();
	int doChunkSize();
	int doReleaseStreamSucc(int event);
	int doFCPublishSucc(int event);
	int doOnFCPublish();
	int doCreateStreamSucc(int event);
	int doPublishSucc();
	int doFCUnPublishSucc();
	int doFCUnPublishResult(int event);
	int doAcknowledgement();
	int doStreamBegin();
	int doPlayReset(std::string instance);
	int doPlayStart(std::string instance);
	int doPlayPublishNotify(std::string instance);
	int doSampleAccess();
	int doStreamDataStart();
	
	void doWriteTimeout();
	void doReadTimeout();

	bool sendPacket(char fmt,const char *timestamp,char *extentimestamp,const char *bodyLen,
		char type,const char *streamId,const char *pData,int len);
	
	CConnRtmp		*msuper;
	cms_timer		*mcmsReadTimeout;
	cms_timer		*mcmsWriteTimeout;
	int				mcmsReadTimeOutDo;
	int				mcmsWriteTimeOutDo;
	LinkUrl			mlinkUrl;
	std::string		murl;
	std::string		mreferUrl;
	std::string		mfcPublishInstance;
	bool			mfinishShake;
	bool			mcomplexShake;
	std::string		mremoteAddr;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;
	CReaderWriter	*mrw;
	RtmpType		mrtmpType;
	RtmpConnStatus	mrtmpStatus;

	//rtmp Э�����
	//read
	bool			misFMLEPublish;//���flashִ�в�ͬ�߼�
	int				mreadChunkSize;
	unsigned int	mreadTotalBytes;
	unsigned int	mreadWindowSize;
	unsigned int	mreadBandWidthSize;
	unsigned int	mreadSequenceNum;
	float			mreadBuffLen;
	char			mreadBandWidthLimit;
	//д
	int				mwriteChunkSize;
	unsigned int	mwriteBandWidthSize;	
	unsigned int	mwirteSequenceNum;
	unsigned int	mwriteTotalBytes;
	unsigned int	mwriteWindowSize;

	char			minBandWidthLimit;
	int				minStreamID;
	int				moutStreamID;
	int				mtransactionID;

	RtmpHeader      mrtmpHeader;
	std::map<int,InboundChunkStream *>  minChunkStreams;
	std::map<int,OutBoundChunkStream *> moutChunkStreams;
	std::map<int,std::string>			mtransIDAction; //rtmp����ID�Ͷ�Ӧ�Ķ���	
	std::map<int,RtmpCommand>			mtransactionCmd;

	c1s1    mc1;
	char	*mps1;

	//����Ƶ���ݽ���3���,�ر�tcp nodelay
	bool			misCloseNodelay;
	unsigned long	mulNodelayEndTime;
};
#endif