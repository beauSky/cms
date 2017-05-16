/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: hsc/kisslovecsh@foxmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
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
#include <strategy/cms_jitter.h>
#include <common/cms_type.h>
#include <protocol/cms_flv_pump.h>
#include <string>

class CRtmpProtocol;
class CFlvTransmission;
class CConnRtmp:public Conn,public CStreamInfo
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
	int doReadData(){return CMS_OK;};
	int doTransmission();
	int sendBefore(const char *data,int len){return 0;};
	void down8upBytes();

	//stream info �ӿ�
	int		firstPlaySkipMilSecond();
	bool	isResetStreamTimestamp();
	bool	isNoTimeout();
	int		liveStreamTimeout();
	int 	noHashTimeout();
	bool	isRealTimeStream();
	int64   cacheTT();
	//std::string getRemoteIP() = 0;
	std::string getHost();
	void    makeOneTask();

	cms_net_ev    *evReadIO();
	cms_net_ev    *evWriteIO();
	
	void setUrl(std::string url);		//�������߱��������߱����ŵĵ�ַ
	void setPushUrl(std::string url);	//���������������������ַ
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
	int	 doRead(bool isTimeout);
	int	 doWrite(bool isTimeout);
	void makeHash();
	void makePushHash();
	void justTick();

	cms_net_ev	*mwatcherReadIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���
	cms_net_ev	*mwatcherWriteIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���

	uint64	mjustTickOld;
	uint64	mjustTick;

	bool		misStop;
	RtmpType	mrtmpType;
	
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

	bool			misPublish;		//�Ƿ��ǿͻ���publish
	bool			misPlay;		//�Ƿ��ǿͻ��˲���
	bool            misPushFlv;		//�Ƿ���flvPoolͶ�ݹ�����
	bool			misPush;		//�Ƿ���push����
	bool			misDown8upBytes;//�Ƿ�ͳ�ƹ�����
	bool			misAddConn;		//�Ƿ������ݵ�����
	//�ٶ�ͳ��
	int32			mxSecdownBytes;
	int32			mxSecUpBytes;
	int32			mxSecTick;

	int64           mllIdx;

	CFlvTransmission *mflvTrans;
	unsigned long  mspeedTick;

	CFlvPump		*mflvPump;
	int64			mcreateTT;

	int64			mtimeoutTick;
};
#endif
