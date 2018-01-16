/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: ���û������/kisslovecsh@foxmail.com

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
/*
��http֧�ּ򵥵�http�����ձ��http-flv����֧��gzip
*/
#ifndef __CMS_HTTP_C_H__
#define __CMS_HTTP_C_H__
#include <interface/cms_interf_conn.h>
#include <interface/cms_read_write.h>
#include <interface/cms_stream_info.h>
#include <protocol/cms_flv_pump.h>
#include <common/cms_type.h>
#include <protocol/cms_http.h>
#include <strategy/cms_jitter.h>
#include <string>

class ChttpClient:public Conn,public CStreamInfo
{
public:
	ChttpClient(HASH &hash,CReaderWriter *rw,std::string pullUrl,std::string oriUrl,
		std::string refer,bool isTls);
	~ChttpClient();
	//conn �ӿ�
	int doit();
	int handleEv(FdEvents *fe);
	int stop(std::string reason);
	std::string getUrl();
	std::string getPushUrl();
	std::string getRemoteIP();
	cms_net_ev    *evReadIO(cms_net_ev *ev = NULL);
	cms_net_ev    *evWriteIO(cms_net_ev *ev = NULL);
	void down8upBytes();

	CReaderWriter *rwConn();

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

	int doDecode();
	int doReadData();
	int doTransmission();
	int sendBefore(const char *data,int len);
	bool isFinish(){return false;};
	void reset(){};

	int doRead(bool isTimeout);
	int doWrite(bool isTimeout);

private:
	int  request();
	int  handle();
	int	 handleFlv(int &ret);
	void makeHash();
	void tryCreateTask();
	int  decodeMetaData(char *data,int len);
	int  decodeVideo(char *data,int len,uint32 timestamp);
	int  decodeAudio(char *data,int len,uint32 timestamp);

	cms_net_ev	*mwatcherReadIO;	
	cms_net_ev	*mwatcherWriteIO;

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

	int   miFirstPlaySkipMilSecond;
	bool  misResetStreamTimestamp;	
	bool  misNoTimeout;
	int   miLiveStreamTimeout;
	int   miNoHashTimeout;
	bool  misRealTimeStream;
	int64 mllCacheTT;

	bool  misChangeMediaInfo;
	bool  misPushFlv;			//�Ƿ���flvPoolͶ�ݹ�����
	bool  misDown8upBytes;		//�Ƿ�ͳ�ƹ�����
	bool  misStop;

	//�ٶ�ͳ��
	int32			mxSecdownBytes;
	int32			mxSecUpBytes;
	int32			mxSecTick;

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

	CFlvPump		*mflvPump;
	int64			mcreateTT;

	int64			mtimeoutTick;
};

#endif
