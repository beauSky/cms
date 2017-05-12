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
#include <protocol/cms_rtmp.h>
#include <protocol/cms_amf0.h>
#include <common/cms_utility.h>
#include <common/cms_char_int.h>
#include <ev/cms_ev.h>
#include <conn/cms_conn_rtmp.h>
#include <log/cms_log.h>
#include <errno.h>
#include <sstream>
#include <arpa/inet.h>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;

#define  MapInIterator  std::map<int,InboundChunkStream *>::iterator 
#define  MapOutIterator std::map<int,OutBoundChunkStream *>::iterator
#define  MapTransactionIterator std::map<int,RtmpCommand>::iterator

#define SHAKE_RAND_DATA_LEN	1536

CRtmpProtocol::CRtmpProtocol(void *super,RtmpType rtmpType,CBufferReader *rd,
							 CBufferWriter *wr, CReaderWriter *rw,std::string remoteAddr)
{
	mremoteAddr = remoteAddr;
	mcmsReadTimeout = NULL;
	mcmsWriteTimeout = NULL;
	msuper = (CConnRtmp *)super;
	mrdBuff = rd;
	mwrBuff = wr;
	mrw = rw;
	mrtmpType = rtmpType;
	mrtmpStatus = RtmpStatusShakeNone;
	mreadChunkSize = DEFAULT_CHUNK_SIZE;
	mreadTotalBytes = 0;
	mreadWindowSize = DEFAULT_WINDOW_SIZE;
	mreadBandWidthSize = DEFAULT_BANDWIDTH_SIZE;
	mreadSequenceNum = 0;
	mreadBuffLen = 0.0;
	mreadBandWidthLimit = 0;
	minBandWidthLimit = 0;
	minStreamID = 0;
	mwriteChunkSize = DEFAULT_CHUNK_SIZE;
	mwriteBandWidthSize = DEFAULT_WINDOW_SIZE;	
	mwirteSequenceNum = 0;
	mwriteTotalBytes = 0;
	mwriteWindowSize = DEFAULT_WINDOW_SIZE;
	mfinishShake = false;
	mcomplexShake = true;
	mtransactionID = 1;
	mcmsReadTimeOutDo = 0;
	mcmsWriteTimeOutDo = 0;
	mrw->setNodelay(1);
	misCloseNodelay = false;
	mulNodelayEndTime = 0;
	mps1 = NULL;
}

CRtmpProtocol::~CRtmpProtocol()
{
	logs->debug("%s [CRtmpProtocol::~CRtmpProtocol] enter %s",
		mremoteAddr.c_str(),getRtmpType().c_str());
	MapInIterator itIn = minChunkStreams.begin();
	for (; itIn != minChunkStreams.end();)
	{
		if (itIn->second != NULL)
		{
			if (itIn->second->lastHeader != NULL)
			{
				delete itIn->second->lastHeader;
			}
			if (itIn->second->currentMessage != NULL)
			{
				if (itIn->second->currentMessage->buffer != NULL)
				{
					delete []itIn->second->currentMessage->buffer;
				}
				itIn->second->currentMessage;
			}
			delete itIn->second;
		}
		minChunkStreams.erase(itIn++);
	}
	MapOutIterator itOut = moutChunkStreams.begin();
	for (; itOut != moutChunkStreams.end();)
	{
		if (itOut->second != NULL)
		{
			if (itOut->second->lastHeader != NULL)
			{
				delete itOut->second->lastHeader;
			}			
			delete itOut->second;
		}
		moutChunkStreams.erase(itOut++);
	}
	if (mps1)
	{
		delete []mps1;
		mps1 = NULL;
	}
	if (mcmsReadTimeout)
	{
		delete mcmsReadTimeout;
	}
	if (mcmsWriteTimeout)
	{
		//��ĵط����ܻ���ʹ�ã�����ֱ��delete
		freeCmsTimer(mcmsWriteTimeout);
	}
	if (mcmsReadTimeout)
	{
		//��ĵط����ܻ���ʹ�ã�����ֱ��delete
		freeCmsTimer(mcmsReadTimeout);
	}
}

bool CRtmpProtocol::run()
{
	doReadTimeout();
	return true;
}

int CRtmpProtocol::handShake()
{
	if ((mrtmpType == RtmpClient2Play || mrtmpType == RtmpClient2Publish))
	{
		if (mrtmpStatus == RtmpStatusShakeNone)
		{
			return c2sComplexShakeC0C1();
		}
		if (mrtmpStatus == RtmpStatusShakeC0C1)
		{
			if (c2sComplexShakeS0() == CMS_ERROR)
			{
				return CMS_ERROR;
			}
		}
		if (mrtmpStatus == RtmpStatusShakeS0)
		{
			if (c2sComplexShakeS1C2() == CMS_ERROR)
			{
				return CMS_ERROR;
			}
		}
		if (mrtmpStatus == RtmpStatusShakeS1)
		{
			if (c2sComplexShakeS2() == CMS_ERROR)
			{
				return CMS_ERROR;
			}
		}
	}
	if (mrtmpType == RtmpServerBPlayOrPublish)
	{
		if (mcomplexShake)
		{
			return s2cComplexShakeC012();
		}
		else
		{
			return s2cSampleShakeC012();
		}
	}
	return CMS_OK;
}

int CRtmpProtocol::c2sComplexShakeC0C1()
{
	logs->debug("%s [CRtmpProtocol::c2sComplexShakeC0C1] enter rtmp %s",
		mremoteAddr.c_str(),getRtmpType().c_str());
	char szC0 = 0x03;//���ֵ�һ�ֽ� 0x03
	int nwrite = 0;
	int ret = mrw->write(&szC0,1,nwrite);
	if (ret != CMS_OK)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s write 0x03 fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
		return CMS_ERROR;
	}
	if (mc1.c1_create(srs_schema1) == -1)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s c1_create fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	char *pc1 = new char[SHAKE_RAND_DATA_LEN];
	mc1.dump(pc1);
	bool isValid;
	ret = mc1.c1_validate_digest(isValid);
	if (ret != 0 || !isValid)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s c1ValidateDigest fail in doComplexShakeC01 ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		delete[] pc1;
		return CMS_ERROR;
	}
	nwrite = 0;
	ret = mrw->write(pc1,SHAKE_RAND_DATA_LEN,nwrite);
	if (ret != CMS_OK || nwrite != SHAKE_RAND_DATA_LEN)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s write c1 fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
		delete[] pc1;
		return CMS_ERROR;
	}
	delete[] pc1;
	mrtmpStatus = RtmpStatusShakeC0C1;
	return CMS_OK;
}

int CRtmpProtocol::c2sComplexShakeS0()
{
	if (mrdBuff->size() < 1 && mrdBuff->grow(1) == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeS0] rtmp %s grow 1 byte fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
		return CMS_ERROR;
	}
	if (mrdBuff->size() < 1)
	{
		//data not enough
		return CMS_OK;
	}
	char ch = mrdBuff->readByte();
	if (ch != 0x03 && ch != 0x06)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s complex shake 0x03 or 06 fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	mrtmpStatus = RtmpStatusShakeS0;
	return CMS_OK;
}

int CRtmpProtocol::c2sComplexShakeS1C2()
{
	if (mrdBuff->size() < SHAKE_RAND_DATA_LEN && mrdBuff->grow(SHAKE_RAND_DATA_LEN-mrdBuff->size()) == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeS1C2] rtmp %s 1 grow bytes fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
		return CMS_ERROR;
	}
	if (mrdBuff->size() < SHAKE_RAND_DATA_LEN)
	{
		//data not enough
		return CMS_OK;
	}
	c1s1 s1;
	int ret = s1.c1_parse(mrdBuff->readBytes(SHAKE_RAND_DATA_LEN),mc1.schema);
	if (ret == -1)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s complex parse s1 fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	c2s2 c2;
	if (c2.c2_create(&s1) == -1)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s complex c2Create fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	char *pc2 = new char[SHAKE_RAND_DATA_LEN];
	c2.dump(pc2);
	int nwrite = 0;
	ret = mrw->write(pc2,SHAKE_RAND_DATA_LEN,nwrite);
	if(ret != CMS_OK || nwrite != SHAKE_RAND_DATA_LEN)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeC0C1] rtmp %s write c2 fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
		delete[] pc2;
		return CMS_ERROR;
	}
	delete[] pc2;
	mrtmpStatus = RtmpStatusShakeS1;
	return CMS_OK;
}

int CRtmpProtocol::c2sComplexShakeS2()
{
	if (mrdBuff->size() < SHAKE_RAND_DATA_LEN && mrdBuff->grow(SHAKE_RAND_DATA_LEN-mrdBuff->size()) == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::c2sComplexShakeS2] rtmp %s 2 grow bytes fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
		return CMS_ERROR;
	}
	if (mrdBuff->size() < SHAKE_RAND_DATA_LEN)
	{
		//data not enough
		return CMS_OK;
	}
	mrdBuff->skip(SHAKE_RAND_DATA_LEN);
	mrtmpStatus = RtmpStatusShakeSuccess;
	mfinishShake = true;
	return doConnect();
}

int CRtmpProtocol::s2cComplexShakeC012()
{
	if (mrtmpStatus == RtmpStatusShakeNone)
	{
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN+1 && mrdBuff->grow(SHAKE_RAND_DATA_LEN+1) == CMS_ERROR)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s 2 grow SHAKE_RAND_DATA_LEN+1 bytes fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
			return CMS_ERROR;
		}
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN+1)
		{
			//data not enough
			return CMS_OK;
		}
		char *ch = mrdBuff->peek(1);
		if (*ch != 0x03 && *ch != 0x06)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s complex shake 0x03 or 06 fail ***",
				mremoteAddr.c_str(),getRtmpType().c_str());
			return CMS_ERROR;
		}
		char *c0 = mrdBuff->peek(SHAKE_RAND_DATA_LEN+1)+1;
		c1s1 c1;
		if ((c1.c1_parse(c0, srs_schema0)) != 0)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s parse c1 schema%d error ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),srs_schema0);
			return CMS_ERROR;
		}
		// try schema1  
		bool isValid = false;  
		if (c1.c1_validate_digest(isValid) != 0 || !isValid)
		{
			if (c1.c1_parse(c0, srs_schema1) != 0)
			{
				logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s parse c1 schema%d error ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),srs_schema1);
				return CMS_ERROR;
			}
			if (c1.c1_validate_digest(isValid) != 0 || !isValid)
			{
				logs->info("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s all schema valid failed, try simple handshake ",
					mremoteAddr.c_str(),getRtmpType().c_str(),srs_schema1);
				mcomplexShake = false;
				return s2cSampleShakeC012();
			}
		}
		mrdBuff->skip(SHAKE_RAND_DATA_LEN+1);
		// encode s1  
		c1s1 s1;  
		if (s1.s1_create(&c1) != 0)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s create s1 from c1 failed ***",
				mremoteAddr.c_str(),getRtmpType().c_str());
			return CMS_ERROR;
		}
		c2s2 s2;  
		if (s2.s2_create(&c1) != 0)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s create s2 from c1 failed ***",
				mremoteAddr.c_str(),getRtmpType().c_str());
			return CMS_ERROR;
		}
		// ���� s0s1s2  
		char* s0s1s2 = new char[3073];
		s0s1s2[0] = 0x03;
		s1.dump(s0s1s2 + 1);
		s2.dump(s0s1s2 + 1537);

		int nwrite = 0;
		int ret = mrw->write(s0s1s2,3073,nwrite);
		if(ret != CMS_OK || nwrite != 3073)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s write s0s1s2 fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
			delete[] s0s1s2;
			return CMS_ERROR;
		}
		delete[] s0s1s2;
		mrtmpStatus = RtmpStatusShakeC0C1;
	}
	else
	{
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN && mrdBuff->grow(SHAKE_RAND_DATA_LEN) == CMS_ERROR)
		{
			logs->error("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s 2 grow SHAKE_RAND_DATA_LEN bytes fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
			return CMS_ERROR;
		}
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN) //���ݻ�û�������
		{
			return CMS_OK;
		}
		mrdBuff->skip(SHAKE_RAND_DATA_LEN);
		logs->debug("%s [CRtmpProtocol::s2cComplexShakeC012] rtmp %s handshake succ.",
			mremoteAddr.c_str(),getRtmpType().c_str());
		mrtmpStatus = RtmpStatusShakeSuccess;
		mfinishShake = true;
	}
	return CMS_OK;
}

int CRtmpProtocol::s2cSampleShakeC012()
{
	if (mrtmpStatus == RtmpStatusShakeNone)
	{
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN+1)
		{
			//data not enough
			return CMS_OK;
		}
		char ch = mrdBuff->readByte();
		if (ch != 0x03 && ch != 0x06)
		{
			logs->error("%s [CRtmpProtocol::s2cSampleShakeC012] rtmp %s complex shake 0x03 or 06 fail ***",
				mremoteAddr.c_str(),getRtmpType().c_str());
			return CMS_ERROR;
		}
		mps1 = new char[SHAKE_RAND_DATA_LEN];
		char *pS0S1S2 = new char[SHAKE_RAND_DATA_LEN+1];
		pS0S1S2[0] = 0x03;
		//s1
		char *pSS = pS0S1S2+1;
		struct timeval tv;
		gettimeofday(&tv,NULL);
		unsigned long ulTime = htonl(tv.tv_sec);
		memcpy(pSS,(char *)&ulTime,4);

		//5-8�ֽڶ�Ϊ 0��Э���޸��ˣ�
		pSS[4] = 0x00;
		pSS[5] = 0x00;
		pSS[6] = 0x00;
		pSS[7] = 0x00;
		//����ַ���
		for (int i = 8; i < SHAKE_RAND_DATA_LEN-8; ++i)
		{
			pSS[i] = rand() % 256;
		}

		int nwrite = 0;
		int ret = mrw->write(pS0S1S2,SHAKE_RAND_DATA_LEN+1,nwrite);
		if(ret != CMS_OK || nwrite != SHAKE_RAND_DATA_LEN+1)
		{
			logs->error("%s [CRtmpProtocol::s2cSampleShakeC012] rtmp %s write s0s1 fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
			delete[] pS0S1S2;
			return CMS_ERROR;
		}
		memcpy(mps1,pSS,SHAKE_RAND_DATA_LEN);
		//s2
		char *c1 = mrdBuff->readBytes(SHAKE_RAND_DATA_LEN);
		//memcpy(pSS,c1+1,4);
		//memcpy(pSS+4,(char *)&ulTime,4);
		//memcpy(pSS+8,c1+9,SHAKE_RAND_DATA_LEN-8);

		nwrite = 0;
		ret = mrw->write(c1,SHAKE_RAND_DATA_LEN,nwrite);
		if(ret != CMS_OK || nwrite != SHAKE_RAND_DATA_LEN)
		{
			logs->error("%s [CRtmpProtocol::s2cSampleShakeC012] rtmp %s write s2 fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrw->errnos(),mrw->errnoCode());
			delete[] pS0S1S2;
			return CMS_ERROR;
		}
		delete[] pS0S1S2;
		mrtmpStatus = RtmpStatusShakeC1;
	}
	else if (mrtmpStatus == RtmpStatusShakeC1)
	{
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN && mrdBuff->grow(SHAKE_RAND_DATA_LEN) == CMS_ERROR)
		{
			logs->error("%s [CRtmpProtocol::s2cSampleShakeC012] rtmp %s 1 grow SHAKE_RAND_DATA_LEN bytes fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
			return CMS_ERROR;
		}
		if (mrdBuff->size() < SHAKE_RAND_DATA_LEN)
		{
			return CMS_OK;
		}
		mrdBuff->skip(SHAKE_RAND_DATA_LEN);
		logs->debug("%s [CRtmpProtocol::s2cSampleShakeC012] rtmp %s handshake succ ",
			mremoteAddr.c_str(),getRtmpType().c_str());
		mrtmpStatus = RtmpStatusShakeSuccess;
		mfinishShake = true;
	}
	return CMS_OK;
}

int CRtmpProtocol::want2Read(bool isTimeout)
{
	if (isTimeout)
	{
		assert(mcmsReadTimeOutDo==1);
		mcmsReadTimeOutDo--;
	}
	if (!mfinishShake)
	{
		if (handShake() == CMS_ERROR)
		{
			return CMS_ERROR;
		}
		if (mfinishShake)
		{
			if (doChunkSize() == CMS_ERROR)
			{
				return CMS_ERROR;
			}
		}
	}
	while (mfinishShake)
	{
		int ret = readMessage();
		if (ret == CMS_ERROR)
		{
			msuper->down8upBytes();
			return CMS_ERROR;
		}
		if (ret == 0)
		{
			msuper->down8upBytes();
			break;
		}
	}
	return CMS_OK;
}

int CRtmpProtocol::want2Write(bool isTimeout)
{
	if (isTimeout)
	{
		assert(mcmsWriteTimeOutDo==1);
		mcmsWriteTimeOutDo--;
	}
	if (!mfinishShake)
	{
		if (handShake() == CMS_ERROR)
		{
			return CMS_ERROR;
		}
		if (mfinishShake)
		{
			if (doChunkSize() == CMS_ERROR)
			{
				return CMS_ERROR;
			}
		}
	}
	int ret = mwrBuff->flush();
	if (ret == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::doRtmpConnect] rtmp %s flush fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}	
	//do write
	if (mrtmpType == RtmpClient2Play || mrtmpType == RtmpServerBPublish || mrtmpType == RtmpServerBPlayOrPublish)
	{
		//logs->debug("%s [CRtmpProtocol::doRtmpConnect] rtmp %s do not have data to send.",
		//	mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_OK;
	}
	else
	{
		if (!mwrBuff->isUsable())
		{
			//���CBufferWriter���п͹۵�����û���ͳ�ȥ��������ʱ��ʱ������ʱ�������ݣ��Ҳ��ٶ�ȡ�κ�����
			//logs->debug("%s [CRtmpProtocol::doRtmpConnect] rtmp %s too many data left",
			//	mremoteAddr.c_str(),getRtmpType().c_str());
			doWriteTimeout();
			return CMS_OK;
		}
		//ֻ�в������ӻ���ת���������Ҫ����������
		if (mrtmpType == RtmpServerBPlay || mrtmpType == RtmpClient2Publish)
		{
			int ret = msuper->doTransmission();
			if (ret < 0)
			{
				return CMS_ERROR;
			}
			if (ret == 0 || ret == 2)
			{
				//logs->debug("%s [CRtmpProtocol::doRtmpConnect] rtmp %s not have buffer",
				//	mremoteAddr.c_str(),getRtmpType().c_str());
				doWriteTimeout();
			}
		}
		msuper->down8upBytes();
	}
	return CMS_OK;
}

int CRtmpProtocol::wait2Read()
{
	return 0;
}

int CRtmpProtocol::wait2Write()
{
	return 0;
}

int CRtmpProtocol::readMessage()
{
	char fmt = 0;
	int  cid = 0;
	int handleLen = 0;
	int ret = readBasicHeader(fmt,cid,handleLen);
	if (ret == CMS_ERROR)
	{
		return CMS_ERROR;
	}
	if (ret == 0)
	{
		return CMS_OK;
	}
	memset(&mrtmpHeader,0,sizeof(mrtmpHeader));
	ret = readRtmpHeader(mrtmpHeader,fmt,handleLen);
	if (ret == CMS_ERROR)
	{
		return CMS_ERROR;
	}
	if (ret == 0)
	{
		return CMS_OK;
	}
	ret = readRtmpPlayload(mrtmpHeader,fmt,cid,handleLen);
	if (ret == CMS_ERROR)
	{
		return CMS_ERROR;
	}
	if (ret == 0)
	{
		return CMS_OK;
	}
	mrdBuff->skip(handleLen);
	return handleLen;
}

int CRtmpProtocol::readBasicHeader(char &fmt,int &cid,int &handleLen)
{
	if ( mrdBuff->size() < 1 && mrdBuff->grow(1) == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::readBasicHeader] rtmp %s grow 1 bytes fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
		return CMS_ERROR;
	}
	if (mrdBuff->size() < 1)
	{
		return CMS_OK;//û��������
	}
	//logs->debug("fmt=%d\n",mrdBuff->peek(handleLen+1)[0] );
	fmt = (mrdBuff->peek(handleLen+1)[0] & 0xC0) >> 6;
	cid = (int)(mrdBuff->peek(handleLen+1)[0] & 0x3F);
	handleLen++;
	if (cid < 0)
	{
		logs->error("%s [CRtmpProtocol::readBasicHeader] rtmp %s cid error ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	switch (cid)
	{
	case 0:
		{
			if ( mrdBuff->size() < 2 && mrdBuff->grow(1) == CMS_ERROR)
			{
				logs->error("%s [CRtmpProtocol::readBasicHeader] 1 rtmp %s grow 1 bytes fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
				return CMS_ERROR;
			}
			if (mrdBuff->size() < 2)
			{
				return CMS_OK;//û��������
			}
			cid = 64 + mrdBuff->peek(handleLen+1)[handleLen];
			handleLen++;
		}
		break;
	case 1:
		{
			if ( mrdBuff->size() < 3 && mrdBuff->grow(2) == CMS_ERROR)
			{
				logs->error("%s [CRtmpProtocol::readBasicHeader] 1 rtmp %s grow 1 bytes fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),mrdBuff->errnos(),mrdBuff->errnoCode());
				return CMS_ERROR;
			}
			if (mrdBuff->size() < 3)
			{
				return CMS_OK;//û��������
			}
			char *u16 = mrdBuff->peek(3)+1;
			cid = 256*u16[1] + u16[0] + 64;
			handleLen += 2;
		}
		break;
	default:
		//logs->error("%s [CRtmpProtocol::readBasicHeader] rtmp %s unknow cid %d ***",
		//	mremoteAddr.c_str(),getRtmpType().c_str(),cid);
		//return CMS_ERROR;
		break;
	}
	return handleLen;
}

int CRtmpProtocol::readRtmpHeader(RtmpHeader &header,int fmt,int &handleLen)
{
	static char rhSizes[] = {11, 7, 3, 0};
	if (fmt > HEADER_FORMAT_CONTINUATION)
	{
		logs->error("%s [CRtmpProtocol::readRtmpHeader] rtmp %s fmt=%d error ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),fmt);
		return CMS_ERROR;
	}
	int size = rhSizes[fmt];
	//printf(">>>>rtmp header fmt=%d, size=%d, handlle=%d\n", fmt,size,handleLen);
	if (fmt == HEADER_FORMAT_CONTINUATION)
	{
		//û��rtmpͷ
		return 1;
	}
	size += handleLen;
	if ( mrdBuff->size() < size && mrdBuff->grow(size-mrdBuff->size()) == CMS_ERROR)
	{
		logs->error("%s [CRtmpProtocol::readRtmpHeader] rtmp %s grow rtmp size %d bytes fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),size,mrdBuff->errnos(),mrdBuff->errnoCode());
		return CMS_ERROR;
	}
	//printf(">>>>rtmp header mrdBuff->size()=%d\n",mrdBuff->size());
	if (mrdBuff->size() < size)
	{
		return 0;//û��������
	}
	int len = 0;
	char *pp = mrdBuff->peek(size)+handleLen;
	if (fmt <= HEADER_FORMAT_SAME_LENGTH_AND_STREAM)
	{
		//timestamp ռ3�ֽ�
		header.timestamp = (unsigned int)bigInt24(pp);
		//printf(">>>>rtmp header timestamp=%u\n",header.timestamp);
		len += 3;
	}
	if (fmt <= HEADER_FORMAT_SAME_STREAM)
	{
		//AMFsize AMFtype 4�ֽ�,timestamp ������һ������
		header.msgLength = (unsigned int)bigInt24(pp+len);
		//printf(">>>>rtmp header msgLength=%u\n",header.msgLength);
		len += 3;
		header.msgTypeID =  (unsigned int)(*(pp+len));
		//printf(">>>>rtmp header msgTypeID=%u\n",header.msgTypeID);
		len += 1;
	}
	if (fmt <= HEADER_FORMAT_FULL)
	{
		//stream id 4�ֽ�
		header.msgStreamID =  (unsigned int)littleInt32(pp+len);
		//printf(">>>>rtmp header msgStreamID=%u\n",header.msgStreamID);
		len += 4;
	}
	if (header.timestamp == TIMESTAMP_EXTENDED)
	{
		size += 4;
		if ( mrdBuff->size() < size && mrdBuff->grow(size-mrdBuff->size()) == CMS_ERROR)
		{
			logs->error("*** %s [CRtmpProtocol::readRtmpHeader] rtmp %s grow rtmp size %d bytes fail,errno=%d,strerrno=%s ***\n",
				mremoteAddr.c_str(),getRtmpType().c_str(),size,mrdBuff->errnos(),mrdBuff->errnoCode());
			return CMS_ERROR;
		}
		if (mrdBuff->size() < size)
		{
			return 0;//û��������
		}
		header.extendedTimestamp = (unsigned int)bigInt32(pp+len);
		len += 4;
	}
	handleLen += len;
	return len;
}

void CRtmpProtocol::copyRtmpHeader(RtmpHeader *pDest,RtmpHeader *pSrc)
{
// 	pDest->format = pSrc->format;
// 	pDest->chunkStreamID = pSrc->chunkStreamID;
	pDest->extendedTimestamp = pSrc->extendedTimestamp;
	pDest->msgLength = pSrc->msgLength;
	pDest->msgStreamID = pSrc->msgStreamID;
	pDest->msgTypeID = pSrc->msgTypeID;
	pDest->timestamp = pSrc->timestamp;
}

int CRtmpProtocol::readRtmpPlayload(RtmpHeader &header,int fmt,int cid,int &handleLen)
{
	MapInIterator itIn = minChunkStreams.find(cid);
	InboundChunkStream *pIncs = NULL;
	bool isFindIncs = false;
	if (itIn == minChunkStreams.end())
	{
		pIncs = new InboundChunkStream;
		memset(pIncs,0,sizeof(InboundChunkStream));
		minChunkStreams.insert(make_pair(cid,pIncs));
	}
	else
	{
		isFindIncs = true;
		pIncs = itIn->second;
	}
	if (pIncs->lastHeader == NULL && fmt != HEADER_FORMAT_FULL)
	{
		logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp first full rtmp head error ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;		
	}
	int dataLen = 0;
	int len = 0;
	int needSkip4Bytes = 0;
	unsigned int streamID = 0;
	unsigned int absoluteTimestamp; //timestamp
	RtmpMessage *pMsg = NULL;
	switch (fmt)
	{
	case HEADER_FORMAT_FULL:
		{
			if (pIncs->lastHeader == NULL)
			{
				pIncs->lastHeader = new RtmpHeader;
			}
			//��ȡ�ð����ݳ���
			dataLen = (int)header.msgLength > mreadChunkSize ? mreadChunkSize : (int)header.msgLength;
			//�ð����Ȳ�����������
			if ( mrdBuff->size() < handleLen+dataLen && mrdBuff->grow(handleLen+dataLen-mrdBuff->size()) == CMS_ERROR)
			{
				logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_FULL,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+dataLen,mrdBuff->errnos(),mrdBuff->errnoCode());
				return CMS_ERROR;
			}
			if (mrdBuff->size() < handleLen+dataLen)
			{
				return 0;
			}
			//�滻��ǰ����ͬ��channelID
			//������timestamp
			copyRtmpHeader(pIncs->lastHeader,&header);
			absoluteTimestamp = header.timestamp != TIMESTAMP_EXTENDED ? header.timestamp : header.extendedTimestamp;
			streamID = header.msgStreamID;
			pMsg = pIncs->currentMessage;
			if (pMsg)
			{
				pMsg->dataLen = 0;
			}
			len = dataLen;
		}
		break;
	case HEADER_FORMAT_SAME_STREAM:
		{
			//��ȡ�ð����ݳ���
			dataLen = (int)header.msgLength > mreadChunkSize ? mreadChunkSize : (int)header.msgLength;
			//�ð����Ȳ�����������
			if ( mrdBuff->size() < handleLen+dataLen && mrdBuff->grow(handleLen+dataLen-mrdBuff->size()) == CMS_ERROR)
			{
				logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_SAME_STREAM,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+dataLen,mrdBuff->errnos(),mrdBuff->errnoCode());
				return CMS_ERROR;
			}
			if (mrdBuff->size() < handleLen+dataLen)
			{
				return 0;
			}
			streamID = pIncs->lastHeader->msgStreamID;
			header.msgStreamID = pIncs->lastHeader->msgStreamID;
			copyRtmpHeader(pIncs->lastHeader,&header);
			if (header.timestamp != TIMESTAMP_EXTENDED)
			{
				absoluteTimestamp = pIncs->lastInAbsoluteTimestamp + header.timestamp;
			}
			else
			{
				absoluteTimestamp = header.extendedTimestamp;
			}
			pMsg = pIncs->currentMessage;
			if (pMsg)
			{
				pMsg->dataLen = 0;
			}
			len = dataLen;
		}
		break;
	case HEADER_FORMAT_SAME_LENGTH_AND_STREAM:
		{
			streamID = pIncs->lastHeader->msgStreamID;
			header.msgStreamID = pIncs->lastHeader->msgStreamID;
			header.msgLength = pIncs->lastHeader->msgLength;
			header.msgTypeID = pIncs->lastHeader->msgTypeID;

			//��ȡ�ð����ݳ���
			dataLen = (int)header.msgLength > mreadChunkSize ? mreadChunkSize : (int)header.msgLength;
			//�ð����Ȳ�����������
			if ( mrdBuff->size() < handleLen+dataLen && mrdBuff->grow(handleLen+dataLen-mrdBuff->size()) == CMS_ERROR)
			{
				logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_SAME_LENGTH_AND_STREAM,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+dataLen,mrdBuff->errnos(),mrdBuff->errnoCode());
				return CMS_ERROR;
			}
			if (mrdBuff->size() < handleLen+dataLen)
			{
				return 0;
			}
			copyRtmpHeader(pIncs->lastHeader,&header);
			if (header.timestamp != TIMESTAMP_EXTENDED)
			{
				absoluteTimestamp = pIncs->lastInAbsoluteTimestamp + header.timestamp;
			}
			else
			{
				absoluteTimestamp = header.extendedTimestamp;
			}
			pMsg = pIncs->currentMessage;
			if (pMsg)
			{
				pMsg->dataLen = 0;
			}
			len = dataLen;
		}
		break;
	case HEADER_FORMAT_CONTINUATION: //���ݷְ�������һ���������ݺ���һ����һ��
		{
			streamID = pIncs->lastHeader->msgStreamID;
			header.msgStreamID = pIncs->lastHeader->msgStreamID;
			header.msgLength = pIncs->lastHeader->msgLength;
			header.msgTypeID = pIncs->lastHeader->msgTypeID;
			header.timestamp = pIncs->lastHeader->timestamp;
			header.extendedTimestamp = pIncs->lastHeader->extendedTimestamp;
			//���message�Ѿ����ڣ�ʹ����
			if (pIncs->currentMessage != NULL)
			{
				if (pIncs->currentMessage->dataLen == 0) //�µ����ݰ�
				{
					if (header.timestamp != TIMESTAMP_EXTENDED)
					{
						absoluteTimestamp = pIncs->lastInAbsoluteTimestamp + header.timestamp;
					}
					else
					{
						absoluteTimestamp = header.extendedTimestamp;
					}
				}
				else
				{
					if (header.timestamp >= TIMESTAMP_EXTENDED) //����ǲ�ֵ����ݰ���������չʱ�䣬�������ֵ����ݰ�Ҳ������չʱ��
					{
						//iHandleLen += 4; //��Ҫ������չʱ��
						//�ð����Ȳ�����������
						if ( mrdBuff->size() < handleLen+4 && mrdBuff->grow(handleLen+4-mrdBuff->size()) == CMS_ERROR)
						{
							logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_CONTINUATION exttendtimestamp,errno=%d,strerrno=%s ***",
								mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+4,mrdBuff->errnos(),mrdBuff->errnoCode());
							return CMS_ERROR;
						}
						if (mrdBuff->size() < handleLen+4)
						{
							return 0;
						}
						unsigned int timestamp = (unsigned int)bigInt32(mrdBuff->peek(handleLen+4)+handleLen);
						if (timestamp == header.extendedTimestamp)
						{
							len += 4; //��Ҫ������չʱ��
							needSkip4Bytes = 4;
						}
						else
						{
							// 								_OUTPUT(SERIOUS,"*** rtmp continuation packet should have extern timestamp,but do not exist,[%u/%u] ***!!!\n",
							// 									m_rtmpHeader.extendedTimestamp,timestamp);
						}
					}
					absoluteTimestamp = pIncs->lastInAbsoluteTimestamp;
				}
				pMsg = pIncs->currentMessage;
				//���㱾�����ݳ���
				unsigned int uiLeft = header.msgLength - pMsg->dataLen;
				dataLen = (int)uiLeft > mreadChunkSize ? mreadChunkSize : (int)uiLeft;
				//�ð����Ȳ�����������
				if ( mrdBuff->size() < handleLen+dataLen+needSkip4Bytes && mrdBuff->grow(handleLen+dataLen+needSkip4Bytes-mrdBuff->size()) == CMS_ERROR)
				{
					logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_CONTINUATION,errno=%d,strerrno=%s ***",
						mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+dataLen+needSkip4Bytes,mrdBuff->errnos(),mrdBuff->errnoCode());
					return CMS_ERROR;
				}
				if (mrdBuff->size() < handleLen+dataLen+needSkip4Bytes)
				{
					return 0;
				}
			}
			else
			{
				if (header.timestamp >= TIMESTAMP_EXTENDED) //����ǲ�ֵ����ݰ���������չʱ�䣬�������ֵ����ݰ�Ҳ������չʱ��
				{
					//iHandleLen += 4; //��Ҫ������չʱ��
					//�ð����Ȳ�����������
					if ( mrdBuff->size() < handleLen+4 && mrdBuff->grow(handleLen+4-mrdBuff->size()) == CMS_ERROR)
					{
						logs->error("%s [CRtmpProtocol::readRtmpPlayload] 2 rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_CONTINUATION exttendtimestamp,errno=%d,strerrno=%s ***",
							mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+4,mrdBuff->errnos(),mrdBuff->errnoCode());
						return CMS_ERROR;
					}
					if (mrdBuff->size() < handleLen+4)
					{
						return 0;
					}
					unsigned int timestamp = (unsigned int)bigInt32(mrdBuff->peek(handleLen+4)+handleLen);
					if (timestamp == header.extendedTimestamp)
					{
						len += 4; //��Ҫ������չʱ��
						needSkip4Bytes = 4;
					}
					else
					{
						// 								_OUTPUT(SERIOUS,"*** rtmp continuation packet should have extern timestamp,but do not exist,[%u/%u] ***!!!\n",
						// 									m_rtmpHeader.extendedTimestamp,timestamp);
					}
				}
				absoluteTimestamp = pIncs->lastInAbsoluteTimestamp;
				//��ȡ�ð����ݳ���
				dataLen = (int)header.msgLength > mreadChunkSize ? mreadChunkSize : (int)header.msgLength;
				//�ð����Ȳ�����������
				if ( mrdBuff->size() < handleLen+dataLen+needSkip4Bytes && mrdBuff->grow(handleLen+dataLen+needSkip4Bytes-mrdBuff->size()) == CMS_ERROR)
				{
					logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s grow rtmp size %d bytes fail in HEADER_FORMAT_FULL,errno=%d,strerrno=%s ***",
						mremoteAddr.c_str(),getRtmpType().c_str(),handleLen+dataLen+needSkip4Bytes,mrdBuff->errnos(),mrdBuff->errnoCode());
					return CMS_ERROR;
				}
				if (mrdBuff->size() < handleLen+dataLen+needSkip4Bytes)
				{
					return 0;
				}
			}
			copyRtmpHeader(pIncs->lastHeader,&header);
		}
		break;
	default:
		logs->error("%s [CRtmpProtocol::readRtmpPlayload] rtmp %s Not except header format=%d ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),fmt);
		return CMS_ERROR;
	}
	pIncs->lastInAbsoluteTimestamp = absoluteTimestamp;
	if (pMsg == NULL)
	{
		pMsg = new RtmpMessage;
		memset(pMsg,0,sizeof(RtmpMessage));
		pMsg->dataLen = 0;
		if (pIncs->lastHeader->msgLength > 0)
		{
			pMsg->bufLen = pIncs->lastHeader->msgLength;
			pMsg->buffer = new char[pMsg->bufLen];
		}
		if (pIncs->currentMessage)
		{
			logs->warn("%s [CRtmpProtocol::readRtmpPlayload] %s rtmp %s current message is NULL but not delete",
				mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());			
			if (pIncs->currentMessage->buffer)
			{
				delete[] pIncs->currentMessage->buffer;
			}
			delete pIncs->currentMessage;
		}
		pIncs->currentMessage = pMsg;
	}
	else if (pMsg->dataLen == 0)
	{
		//���¿��ٿռ�
		if (pIncs->lastHeader->msgLength > 0 && pMsg->bufLen < pIncs->lastHeader->msgLength)
		{
			delete []pMsg->buffer;
			pMsg->bufLen = pIncs->lastHeader->msgLength;
			pMsg->buffer = new char[pMsg->bufLen];
		}
	}
	memcpy(pMsg->buffer+pMsg->dataLen,mrdBuff->peek(handleLen+needSkip4Bytes+dataLen)+handleLen+needSkip4Bytes,dataLen);
	pMsg->msgType = header.msgTypeID;
	//pMsg->streamId =  header.chunkStreamID;
	pMsg->timestamp = header.timestamp >= TIMESTAMP_EXTENDED ? header.extendedTimestamp : header.timestamp;
	pMsg->absoluteTimestamp = absoluteTimestamp;
	pMsg->dataLen += dataLen;
	handleLen += needSkip4Bytes+dataLen;	
	pMsg->streamId = streamID;
// 	printf(">>>>>pMsg->dataLen=%d,header.msgLength=%d\n",pMsg->dataLen,header.msgLength);
	mreadTotalBytes += handleLen;
	mreadSequenceNum += handleLen;
	if (mreadSequenceNum >= mreadBandWidthSize)
	{
		if (doAcknowledgement() == CMS_ERROR)
		{
			return CMS_ERROR;
		}
		mreadSequenceNum = 0;
	}

	//�ð����������ѽ������
	if (pMsg->dataLen == header.msgLength)
	{
		if (decodeMessage(pMsg) == CMS_OK)
		{
			pMsg->dataLen = 0;
			pMsg->buffer = NULL;
			pMsg->bufLen = 0;
		}
		else
		{
			return CMS_ERROR;
		}
	}
	return handleLen;
}

int  CRtmpProtocol::decodeMessage(RtmpMessage *msg)
{
	return msuper->decodeMessage(msg);
}

int CRtmpProtocol::decodeChunkSize(RtmpMessage *msg)
{
	int *chunkSize = (int *)msg->buffer;
	mreadChunkSize = ntohl(*chunkSize);
	logs->debug("#####%s [CRtmpProtocol::decodeChunkSize] rtmp %s peer chunk size=%d",
		mremoteAddr.c_str(),getRtmpType().c_str(),mreadChunkSize);
	return CMS_OK;
}

int CRtmpProtocol::decodeWindowSize(RtmpMessage *msg)
{
	if (msg->dataLen < 4)
	{
		logs->error("%s [CRtmpProtocol::decodeWindowSize] rtmp %s msg len=%d errro ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),msg->dataLen);
		return CMS_ERROR;
	}
	int *pWindowSize = (int *)msg->buffer;
	mreadWindowSize = ntohl(*pWindowSize);
	return CMS_OK;
}

int CRtmpProtocol::decodeBandWidth(RtmpMessage *msg)
{
	if (msg->dataLen < 4)
	{
		logs->error("%s [CRtmpProtocol::decodeBandWidth] rtmp %s msg len=%d errro ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),msg->dataLen);
		return CMS_ERROR;
	}
	int *pBandWidth = (int *)msg->buffer;
	mreadBandWidthSize = ntohl(*pBandWidth);
	logs->debug("#####%s [CRtmpProtocol::decodeChunkSize] rtmp %s peer bandwidth size=%d",
		mremoteAddr.c_str(),getRtmpType().c_str(),mreadBandWidthSize);
	if (msg->dataLen > 4)
	{
		mreadBandWidthLimit = *(msg->buffer+4);
	}
	return CMS_OK;
}


int CRtmpProtocol::handleUserControlMsg(RtmpMessage *msg)
{
	int ret = CMS_OK;
	short eventType = *((short *)msg->buffer);
	eventType = ntohs(eventType);
	switch (eventType)
	{
	case USER_CONTROL_STREAM_BEGIN:
		{
			minStreamID = *((int *)(msg->buffer + sizeof(short)));
			minStreamID = ntohl(minStreamID);
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream Stream Begin control,stream id=%d",
				mremoteAddr.c_str(),getRtmpType().c_str(),minStreamID);
		}
		break;
	case USER_CONTROL_STREAM_EOF:
		{
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream eof control,discarding.",
				mremoteAddr.c_str(),getRtmpType().c_str());
		}
		break;
	case USER_CONTROL_STREAM_DRY:
		{
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream dry control,discarding.",
				mremoteAddr.c_str(),getRtmpType().c_str());
		}
		break;
	case USER_CONTROL_SET_BUFFER_LENGTH:
		{				
			char *bit = msg->buffer + sizeof(short);
			bit = amf0::bit64Reversal(bit);
			mreadBuffLen = *((float *)bit);
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream set buf len control,buffer len=%f",
				mremoteAddr.c_str(),getRtmpType().c_str(),mreadBuffLen);
		}
		break;
	case USER_CONTROL_STREAM_LS_RECORDED:
		{
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream ls control,discarding.",
				mremoteAddr.c_str(),getRtmpType().c_str());
		}
		break;
	case USER_CONTROL_PING_REQUEST:
		{
			string strRtmpPingRsp;
			char fmt = CHUNK_STREAM_ID_PROTOCOL;
			char timestamp[3] = {0};
			int iBodyLen = 8;
			iBodyLen = htonl(iBodyLen);
			int iStreamID = minStreamID;
			strRtmpPingRsp.append(msg->buffer+sizeof(short),4);

			if (sendPacket(fmt,timestamp,NULL,(const char *)((char*)&iBodyLen)+1,
				MESSAGE_TYPE_USER_CONTROL,(const char *)&iStreamID,msg->buffer+sizeof(short),4) == false)
			{
				logs->error("*** %s [CRtmpProtocol::doRtmpConnect] rtmp %s send rtmp ping response control fail,errno=%d,strerrno=%s ***",
					mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
				ret = CMS_ERROR;
			}
		}
		break;
	case USER_CONTROL_PING_RESPONSE:
		{
			logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received stream ping response control,discarding.",
				mremoteAddr.c_str(),getRtmpType().c_str());
		}
		break;
	default:
		logs->debug("#####%s [CRtmpProtocol::handleUserControlMsg] rtmp %s received unknow stream control[ %d ],discarding.",
			mremoteAddr.c_str(),getRtmpType().c_str(),eventType);
		break;
	}
	return ret;
}

int CRtmpProtocol::decodeAmf03(RtmpMessage *msg,bool isAmf3)
{
	amf0::Amf0Block *block = NULL;
	if (isAmf3)
	{
		block = amf0::amf0Parse(msg->buffer+1,msg->dataLen-1);	
	}
	else
	{
		block = amf0::amf0Parse(msg->buffer,msg->dataLen);	
	}
	if (block == NULL)
	{
		logs->error("*** %s [CRtmpProtocol::decodeAmf03] rtmp %s amf0Parse fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		return CMS_ERROR;
	}
	int ret = CMS_OK;
	if (block->cmd == Amf0CommandResult)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s result ",
			mremoteAddr.c_str(),getRtmpType().c_str());
		string strEvent;
		amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
		if (type == amf0::AMF0_TYPE_NUMERIC)
		{			
			int event = atoi(strEvent.c_str());
			logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s result event %d",
				mremoteAddr.c_str(),getRtmpType().c_str(),event);
			MapTransactionIterator it = mtransactionCmd.find(event);
			if (it != mtransactionCmd.end())
			{
				logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s result find event %d",
					mremoteAddr.c_str(),getRtmpType().c_str(),event);
				ret = decodeCommandResult(it->second);
				mtransactionCmd.erase(it);
			}
		}
	}
	else if (block->cmd == Amf0CommandError)
	{
		logs->error("*** %s [CRtmpProtocol::decodeAmf03] rtmp %s recv _error command ***",
			mremoteAddr.c_str(),getRtmpType().c_str());
		mrtmpStatus = RtmpStatusError;
		ret = CMS_ERROR;
	}
	else if (block->cmd == Amf0CommandConnect)
	{
		logs->debug("### %s [CRtmpProtocol::decodeAmf03] rtmp %s recv connect command ###",
			mremoteAddr.c_str(),getRtmpType().c_str());
		ret = decodeConnect(block);
	}
	else if (block->cmd == Amf0CommandReleaseStream)
	{
		ret = decodeReleaseStream(block);		
	}
	else if (block->cmd == Amf0CommandFcPublish)
	{
		misFMLEPublish = true;
		ret = decodeFcPublish(block);
	}
	else if (block->cmd == Amf0CommandCreateStream)
	{
		ret = decodeCreateStream(block);
	}
	else if (block->cmd == Amf0CommandPublish)
	{	
		moutStreamID = msg->streamId;
		ret = decodePublish(block);
		mrtmpType = RtmpServerBPublish;
		ret = msuper->setPublishTask();
	}
	else if (block->cmd == Amf0CommandPlay)
	{
		mrtmpType = RtmpServerBPlay;
		ret = decodePlay(block);
	}
	else if (block->cmd == "play2")
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 play2 command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}	
	else if (block->cmd == Amf0CommandPause)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 pause command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}
	else if (block->cmd == Amf0CommandDeleteStream)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 deleteStream command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}
	else if (block->cmd == Amf0CommandUnpublish)//ȡ������
	{
		ret = decodeUnPublish(block);
	}
	else if (block->cmd == Amf0CommandCloseStream)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 closeStream command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}
	else if (block->cmd == Amf0CommandOnStatus)
	{
		std::string strCodeValue;
		amf0::amf0Block5Value(block,StatusCode,strCodeValue);
		if (strCodeValue == StatusCodeStreamFailed || strCodeValue == StatusCodeStreamStreamNotFound)
		{
			logs->error("*** %s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 closeStream command code=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),strCodeValue.c_str());
			mrtmpStatus = RtmpStatusError;
			ret = CMS_ERROR;
		}
		else if (strCodeValue == Amf0CommandOnStatus)
		{
			//������
			int ret = msuper->doTransmission();
			if (ret < 0)
			{
				ret = CMS_ERROR;
			}
			else
			{
				mulNodelayEndTime = getTickCount();
				if (ret == 0 || ret == 2)
				{
					if (ret == 2)
					{
						ret = CMS_ERROR;
					}
					else
					{
						ret = CMS_OK;
						doWriteTimeout();
					}
				}				
			}
		}
	}
	else if (block->cmd == Amf0DataSampleAccess)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 RtmpSampleAccess command,discarding,metaData is comming",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}
	else if (block->cmd == Amf0SetDataFrame)//metadata?
	{
		msuper->decodeSetDataFrame(block);
	}				
	else if (block->cmd == Amf0MetaData)
	{
		msuper->decodeMetaData(block);
	}
	else if (block->cmd == Amf0CommandOnBwDone)
	{
		//ĳЩ����������Ҫ������ onBWDone �Ļ�Ӧ���Ż���������ߣ������Ͽ�����
		ret = doCheckBW();
	}
	else if (block->cmd == Amf0MetaCheckbw)
	{
		logs->debug("%s [CRtmpProtocol::decodeAmf03] rtmp %s received amf0 _checkbw command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str());
	}
	else
	{
		logs->error("%s [CRtmpProtocol::decodeAmf03] rtmp %s received unknow amf0 %s command,discarding",
			mremoteAddr.c_str(),getRtmpType().c_str(),block->cmd.c_str());
	}
	amf0::amf0BlockRelease(block);
	return ret;
}

int CRtmpProtocol::decodeCommandResult(RtmpCommand cmd)
{
	int ret = CMS_OK;
	switch (cmd)
	{
	case RtmpCommandConnect:
		{
			if (mrtmpType == RtmpClient2Play)
			{
				if (doCreateStream() == CMS_ERROR)
				{
					ret = CMS_ERROR;
				}
			}
			else if (mrtmpType == RtmpClient2Publish)
			{
				if (doReleaseStream() == CMS_ERROR || 
					doFCPublish() == CMS_ERROR ||
					doCreateStream() == CMS_ERROR)
				{
					ret = CMS_ERROR;
				}
			}
			else
			{
				logs->error("%s [CRtmpProtocol::decodeCommandResult] rtmp %s unexpect cmd",
					mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
				ret = CMS_ERROR;
			}
		}
		break;
	case RtmpCommandCreateStream:
		{
			if (mrtmpType == RtmpClient2Play)
			{
				if (doPlay() == CMS_ERROR || 
					doSetBufLength() == CMS_ERROR )
				{
					ret = CMS_ERROR;
				}
			}
			else if (mrtmpType == RtmpClient2Publish)
			{
				ret = doPublish();
			}
		}
		break;
	case RtmpCommandPlay:
		{

		}
		break;
	case RtmpCommandReleaseStream:
		{

		}
		break;
	case RtmpCommandPublish:
		{

		}
		break;
	case RtmpCommandDeleteStream:
		{

		}
		break;
	default:
		break;
	}
	return ret;
}

int CRtmpProtocol::doConnect()
{
	if (mrtmpType == RtmpClient2Play &&
		msuper->setPlayTask() == CMS_ERROR)
	{
		//���������ж��Ƿ��Ѿ�����
		return CMS_ERROR;
	}
	if (mrtmpType == RtmpClient2Play)
	{
		if (!parseUrl(msuper->getUrl(),mlinkUrl))
		{
			logs->error("*** %s [CRtmpProtocol::doConnect] rtmp %s parse url fail %s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
			return CMS_ERROR;
		}
	}
	else if (mrtmpType == RtmpClient2Publish)
	{
		if (!parseUrl(msuper->getPushUrl(),mlinkUrl))
		{
			logs->error("*** %s [CRtmpProtocol::doConnect] rtmp %s parse url fail %s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
			return CMS_ERROR;
		}
	}
	std::ostringstream strStream;
	if (mlinkUrl.isDefault)
	{
		strStream << mlinkUrl.protocol << "://" << mlinkUrl.host << "/" << mlinkUrl.app;
	}
	else
	{
		strStream << mlinkUrl.protocol << "://" << mlinkUrl.host << ":" << mlinkUrl.port << "/" << mlinkUrl.app;
	}
	logs->info(">>> %s [CRtmpProtocol::doConnect] rtmp %s connect %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandConnect,strlen(Amf0CommandConnect));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,"app",amf0::amf0StringNew((amf0::uint8 *)mlinkUrl.app.c_str(),mlinkUrl.app.length()));
	amf0::amf0ObjectAdd(object,"flashVer",amf0::amf0StringNew((amf0::uint8 *)APP_VERSION,strlen(APP_VERSION)));
	amf0::amf0ObjectAdd(object,"tcUrl",amf0::amf0StringNew((amf0::uint8 *)strStream.str().c_str(),strStream.str().length()));
	amf0::amf0ObjectAdd(object,"fpad",amf0::amf0BooleanNew(0x00));
	amf0::amf0ObjectAdd(object,"capabilities",amf0::amf0NumberNew(15));
	amf0::amf0ObjectAdd(object,"audioCodecs",amf0::amf0NumberNew(4071));
	amf0::amf0ObjectAdd(object,"videoCodecs",amf0::amf0NumberNew(252));
	amf0::amf0ObjectAdd(object,"videoFunction",amf0::amf0NumberNew(0x01));	
	amf0::amf0BlockPush(block,object);
	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doConnect] rtmp %s send connect fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mrtmpStatus = RtmpStatusConnect;
	mtransactionCmd.insert(make_pair(mtransactionID++,RtmpCommandConnect));
	return CMS_OK;
}

int CRtmpProtocol::doReleaseStream()
{
	logs->info(">>> %s [CRtmpProtocol::doConnect] rtmp %s releaseStream %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandReleaseStream,strlen(Amf0CommandReleaseStream));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0StringNew((amf0::uint8 *)mlinkUrl.instanceName.c_str(),mlinkUrl.instanceName.length());
	amf0::amf0BlockPush(block,data);
	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doReleaseStream] rtmp %s send releasestream fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mtransactionID++;
	return CMS_OK;
}

int CRtmpProtocol::doFCPublish()
{
	logs->info(">>> %s [CRtmpProtocol::doConnect] rtmp %s fcPublish %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandFcPublish,strlen(Amf0CommandFcPublish));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0StringNew((amf0::uint8 *)mlinkUrl.instanceName.c_str(),mlinkUrl.instanceName.length());
	amf0::amf0BlockPush(block,data);
	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doFCPublish] rtmp %s send fcpublish fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mtransactionID++;
	return CMS_OK;
}

int CRtmpProtocol::doCreateStream()
{
	logs->info(">>> %s [CRtmpProtocol::doConnect] rtmp %s createStream %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandCreateStream,strlen(Amf0CommandCreateStream));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doCreateStream] rtmp %s send create stream fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mrtmpStatus = RtmpStatusCreateStream;
	mtransactionCmd.insert(make_pair(mtransactionID++,RtmpCommandCreateStream));
	return CMS_OK;
}

int CRtmpProtocol::doPlay()
{
	logs->info(">>> %s [CRtmpProtocol::doPlay] rtmp %s play %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandPlay,strlen(Amf0CommandPlay));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0StringNew((amf0::uint8 *)mlinkUrl.instanceName.c_str(),mlinkUrl.instanceName.length());
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(-2);
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlay] rtmp %s send play fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mrtmpStatus = RtmpStatusPlay1;
	mtransactionCmd.insert(make_pair(mtransactionID++,RtmpCommandPlay));
	return CMS_OK;
}

int CRtmpProtocol::doPublish() 
{
	logs->info(">>> %s [CRtmpProtocol::doPublish] rtmp %s publish %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getPushUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandPublish,strlen(Amf0CommandPublish));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);
	string instanceName = mlinkUrl.instanceName;
	if (instanceName.find("?") == string::npos)
	{
		instanceName += "?belong=cms";
	}
	else if (instanceName.find("belong=cms") == string::npos)
	{
		instanceName += "&belong=cms";
	}
	data = amf0::amf0StringNew((amf0::uint8 *)instanceName.c_str(),instanceName.length());
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0StringNew((amf0::uint8 *)mlinkUrl.app.c_str(),mlinkUrl.app.length());
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPublish] rtmp %s send publish fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mrtmpStatus = RtmpStatusPublish;
	mtransactionCmd.insert(make_pair(mtransactionID++,RtmpCommandPublish));
	return CMS_OK;
}

int CRtmpProtocol::doSetBufLength()
{
	logs->info(">>> %s [CRtmpProtocol::doSetBufLength] rtmp %s set buffer length %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	string strRtmpBody;
	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	int iBodyLen = 10;
	iBodyLen = htonl(iBodyLen);
	short iSetBuffLenCmd = htons(USER_CONTROL_SET_BUFFER_LENGTH);
	int iBuffMinSeconds = htonl(1000);
	strRtmpBody.append((char*)&iSetBuffLenCmd,2);
	int iStreamID = minStreamID;
	strRtmpBody.append((char*)&iStreamID,4);
	strRtmpBody.append((char*)&iBuffMinSeconds,4);
	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_USER_CONTROL,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlay] rtmp %s send set buffer length fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doCheckBW()
{	
	logs->info(">>> %s [CRtmpProtocol::doCheckBW] rtmp %s _checkbw %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)"_checkbw",8);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(mtransactionID);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);
	string strRtmpBody = amf0::amf0Block2String(block);

	amf0::amf0BlockRelease(block);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlay] rtmp %s send _checkbw fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	mtransactionID++;
	return CMS_OK;
}

int CRtmpProtocol::doWindowSize()
{
	logs->info(">>> %s [CRtmpProtocol::doWindowSize] rtmp %s window size %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	string strRtmpBody;

	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	int iBodyLen = 4;

	iBodyLen = htonl(iBodyLen);
	int iWindowSize = htonl(mwriteWindowSize);
	strRtmpBody.append((char *)&iWindowSize,4);
	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_WINDOW_SIZE,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doWindowSize] rtmp %s send window size fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doBandWidth()
{
	logs->info(">>> %s [CRtmpProtocol::doBandWidth] rtmp %s bandwidth %s",
		mremoteAddr.c_str(),getRtmpType().c_str(),msuper->getUrl().c_str());
	string strRtmpBody;
	
	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	int iBodyLen = 5;
	iBodyLen = htonl(iBodyLen);
	int iBandWidth = htonl(mwriteBandWidthSize);
	strRtmpBody.append((char *)&iBandWidth,4);
	strRtmpBody.append(1,0x02);
	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_BANDWIDTH,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doBandWidth] rtmp %s send bandwidth fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doAcknowledgement()
{
	string strRtmpBody;
	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	int iBodyLen = 4;
	iBodyLen = htonl(iBodyLen);
	int seqNo = htonl(mreadTotalBytes);
	strRtmpBody.append((char *)&seqNo,4);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_ACK,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doAcknowledgement] rtmp %s send connect doAcknowledgement fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doConnectSucc(int event,int objectEncoding)
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandResult,strlen(Amf0CommandResult));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(event);
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,"fmsver",amf0::amf0StringNew((amf0::uint8*)APP_VERSION,strlen(APP_VERSION)));
	amf0::amf0ObjectAdd(object,"capabilities",amf0::amf0NumberNew(255));
	amf0::amf0ObjectAdd(object,"mode",amf0::amf0NumberNew(1));
	amf0::amf0BlockPush(block,object);

	object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusLevel,amf0::amf0StringNew((amf0::uint8*)StatusLevelStatus,strlen(StatusLevelStatus)));
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeConnectSuccess,strlen(StatusCodeConnectSuccess)));
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8*)DescriptionConnectionSucceeded,strlen(DescriptionConnectionSucceeded)));
	amf0::amf0ObjectAdd(object,"objectEncoding",amf0::amf0NumberNew(objectEncoding));	

	amf0::Amf0Data *ecmaArray = amf0::amf0EcmaArrayNew();
	amf0::amf0ObjectAdd(ecmaArray,"version",amf0::amf0StringNew((amf0::uint8*)NUM_VERSION,strlen(NUM_VERSION)));
	amf0::amf0ObjectAdd(object,"data",ecmaArray);

	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doConnectSucc] rtmp %s send connect success fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}

	mrtmpStatus = RtmpStatusConnect;
	return CMS_OK;
}

int CRtmpProtocol::doOnBWDone() 
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnBwDone,strlen(Amf0CommandOnBwDone));
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);
	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doOnBWDone] rtmp %s send connect doOnBWDone fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doChunkSize()
{
	string strRtmpBody;
	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};
	mwriteChunkSize = DEFAULT_RTMP_CHUNK_SIZE;
	int iBodyLen = 4;
	iBodyLen = htonl(iBodyLen);
	int iChunkSize = htonl(mwriteChunkSize);
	strRtmpBody.append((char *)&iChunkSize,4);
	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_CHUNK_SIZE,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doChunkSize] rtmp %s send connect doChunkSize fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doReleaseStreamSucc(int event)
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandResult,strlen(Amf0CommandResult));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(event);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0UndefinedNew();
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doReleaseStreamSucc] rtmp %s send connect doReleaseStreamSucc fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doFCPublishSucc(int event)
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandResult,strlen(Amf0CommandResult));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(event);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0UndefinedNew();
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doFCPublishSucc] rtmp %s send connect doFCPublishSucc fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doOnFCPublish()
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnFcPublish,strlen(Amf0CommandOnFcPublish));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodePublishStart,strlen(StatusCodePublishStart)));
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8 *)"Started publishing stream.",strlen("Started publishing stream.")));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_OVERSTREAM;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doOnFCPublish] rtmp %s send connect doOnFCPublish fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doCreateStreamSucc(int event)
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandResult,strlen(Amf0CommandResult));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(event);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(1);
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_COMMAND;
	char timestamp[3] = {0};
	char streamID[4] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doCreateStreamSucc] rtmp %s send connect doCreateStreamSucc fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	if (mrtmpStatus != RtmpStatusPublish)
	{
		mrtmpStatus = RtmpStatusCreateStream;
	}
	return CMS_OK;
}

int CRtmpProtocol::doPublishSucc()
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnStatus,strlen(Amf0CommandOnStatus));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusLevel,amf0::amf0StringNew((amf0::uint8*)StatusLevelStatus,strlen(StatusLevelStatus)));
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodePublishStart,strlen(StatusCodePublishStart)));
	string strDesc = mfcPublishInstance + " is now published.";
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8*)strDesc.c_str(),strDesc.length()));
	amf0::amf0ObjectAdd(object,StatusDetails,amf0::amf0StringNew((amf0::uint8*)mfcPublishInstance.c_str(),mfcPublishInstance.length()));
	amf0::amf0ObjectAdd(object,StatusClientId,amf0::amf0StringNew((amf0::uint8*)"0",1));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPublishSucc] rtmp %s send connect doPublishSucc fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doFCUnPublishSucc() 
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnFcUnpublish,strlen(Amf0CommandOnFcUnpublish));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeUnpublishSuccess,strlen(StatusCodeUnpublishSuccess)));
	string strDesc = "Stop publishing stream.";
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8*)strDesc.c_str(),strDesc.length()));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doFCUnPublishSucc] rtmp %s send connect doFCUnPublishSucc fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doFCUnPublishResult(int event)
{
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandResult,strlen(Amf0CommandResult));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_OVERSTREAM;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doFCUnPublishResult] rtmp %s send connect doFCUnPublishResult fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doStreamBegin()
{
	logs->info("#####%s [CRtmpProtocol::doStreamBegin] rtmp %s doStreamBegin #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	string strRtmpBody;
	char fmt = CHUNK_STREAM_ID_PROTOCOL;
	char timestamp[3] = {0};
	char streamID[4] = {0};
	int iBodyLen = 6;
	iBodyLen = htonl(iBodyLen);
	strRtmpBody.append(2,0x00);
	int iStreamId = htonl(moutStreamID);
	strRtmpBody.append((char *)&iStreamId,4);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_USER_CONTROL,
		(const char*)streamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doStreamBegin] rtmp %s send doStreamBegin fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doPlayReset(std::string instance)
{
	logs->info("#####%s [CRtmpProtocol::doPlayReset] rtmp %s doPlayReset #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnStatus,strlen(Amf0CommandOnStatus));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusLevel,amf0::amf0StringNew((amf0::uint8*)StatusLevelStatus,strlen(StatusLevelStatus)));
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeStreamReset,strlen(StatusCodeStreamReset)));
	std::ostringstream strStream;

	strStream << "Playing and resetting " << instance;
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8 *)strStream.str().c_str(),strStream.str().length()));
	amf0::amf0ObjectAdd(object,StatusDetails,amf0::amf0StringNew((amf0::uint8 *)instance.c_str(),instance.length()));
	amf0::amf0ObjectAdd(object,StatusClientId,amf0::amf0StringNew((amf0::uint8 *)"0",1));

	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlayReset] rtmp %s send doPlayReset fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doPlayStart(std::string instance)
{
	logs->info("#####%s [CRtmpProtocol::doPlayStart] rtmp %s doRtmpPlayStart #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnStatus,strlen(Amf0CommandOnStatus));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusLevel,amf0::amf0StringNew((amf0::uint8*)StatusLevelStatus,strlen(StatusLevelStatus)));
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeStreamStart,strlen(StatusCodeStreamStart)));
	std::ostringstream strStream;
	strStream << "Started playing " << instance;
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8 *)strStream.str().c_str(),strStream.str().length()));
	amf0::amf0ObjectAdd(object,StatusDetails,amf0::amf0StringNew((amf0::uint8 *)instance.c_str(),instance.length()));
	amf0::amf0ObjectAdd(object,StatusClientId,amf0::amf0StringNew((amf0::uint8 *)"0",1));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlayStart] rtmp %s send doPlayStart fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doPlayPublishNotify(std::string instance)
{
	logs->info("#####%s [CRtmpProtocol::doPlayPublishNotify] rtmp %s doPlayPublishNotify #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnStatus,strlen(Amf0CommandOnStatus));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NumberNew(0);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0NullNew();
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusLevel,amf0::amf0StringNew((amf0::uint8*)StatusLevelStatus,strlen(StatusLevelStatus)));
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeStreamPublishNotify,strlen(StatusCodeStreamPublishNotify)));
	std::ostringstream strStream;
	strStream << instance << " is now published.";	
	amf0::amf0ObjectAdd(object,StatusDescription,amf0::amf0StringNew((amf0::uint8 *)strStream.str().c_str(),strStream.str().length()));
	amf0::amf0ObjectAdd(object,StatusDetails,amf0::amf0StringNew((amf0::uint8 *)instance.c_str(),instance.length()));
	amf0::amf0ObjectAdd(object,StatusClientId,amf0::amf0StringNew((amf0::uint8 *)"0",1));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_AMF0,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doPlayPublishNotify] rtmp %s send doPlayPublishNotify fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doSampleAccess()
{
	logs->info("#####%s [CRtmpProtocol::doSampleAccess] rtmp %s doSampleAccess #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0DataSampleAccess,strlen(Amf0DataSampleAccess));
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0BooleanNew(1);
	amf0::amf0BlockPush(block,data);

	data = amf0::amf0BooleanNew(1);
	amf0::amf0BlockPush(block,data);	

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_INVOKE,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doSampleAccess] rtmp %s send doSampleAccess fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::doStreamDataStart()
{
	logs->info("#####%s [CRtmpProtocol::doStreamDataStart] rtmp %s doStreamDataStart #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	amf0::Amf0Block *block = amf0::amf0BlockNew();
	amf0::Amf0Data *data = amf0::amf0StringNew((amf0::uint8 *)Amf0CommandOnStatus,strlen(Amf0CommandOnStatus));
	amf0::amf0BlockPush(block,data);

	amf0::Amf0Data *object = amf0::amf0ObjectNew();
	amf0::amf0ObjectAdd(object,StatusCode,amf0::amf0StringNew((amf0::uint8*)StatusCodeDataStart,strlen(StatusCodeDataStart)));
	amf0::amf0BlockPush(block,object);

	std::string strRtmpBody = amf0::amf0Block2String(block);
	int iBodyLen = amf0::amf0BlockSize(block);
	iBodyLen = htonl(iBodyLen);

	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};

	amf0::amf0BlockRelease(block);

	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_INVOKE,
		(const char*)&moutStreamID,strRtmpBody.c_str(),strRtmpBody.length()) == false)
	{
		logs->error("*** %s [CRtmpProtocol::doStreamDataStart] rtmp %s send doStreamDataStart fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::decodeConnect(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodeConnect] rtmp %s decodeConnect #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	int ret = CMS_OK;
	string strObjectEncoding;
	string strFlashVer;
	int objectEncoding = 0;
	
	amf0::amf0Block5Value(block,"tcUrl",murl);
	amf0::amf0Block5Value(block,"pageUrl",mreferUrl);
	amf0::amf0Block5Value(block,"flashVer",strFlashVer);
	amf0::amf0Block5Value(block,"objectEncoding",strObjectEncoding);
	string strEvent;
	amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
	if (type == amf0::AMF0_TYPE_NUMERIC)
	{
		int event = atoi(strEvent.c_str());
		objectEncoding = atoi(strObjectEncoding.c_str());
		if (doWindowSize() == CMS_ERROR || 
			doBandWidth() == CMS_ERROR ||
			doConnectSucc(event,objectEncoding) == CMS_ERROR ||
			doOnBWDone() == CMS_ERROR)
		{
			ret = CMS_ERROR;
		}
	}
	else
	{
		ret = CMS_ERROR;
	}
	return ret;
}

int CRtmpProtocol::decodeReleaseStream(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodeReleaseStream] rtmp %s releaseStream #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	int ret = CMS_OK;
	std::string strInstanceName;
	amf0::amf0Block5Value(block,3,strInstanceName);
	string strEvent;
	amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
	if (type == amf0::AMF0_TYPE_NUMERIC)
	{
		int event = atoi(strEvent.c_str());
		ret = doReleaseStreamSucc(event);
	}
	return ret;
}

int CRtmpProtocol::decodeFcPublish(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodeFcPublish] rtmp %s fcPublish #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	int ret = CMS_OK;
	amf0::amf0Block5Value(block,3,mfcPublishInstance);
	string strEvent;
	amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
	if (type == amf0::AMF0_TYPE_NUMERIC)
	{
		int event = atoi(strEvent.c_str());
		ret = doFCPublishSucc(event);
	}
	return ret;
}

int CRtmpProtocol::decodeUnPublish(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodeUnPublish] rtmp %s unPublish #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	int ret = CMS_OK;
	string strEvent;
	amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
	if (type == amf0::AMF0_TYPE_NUMERIC)
	{
		int event = atoi(strEvent.c_str());
		if (doFCUnPublishSucc() == CMS_ERROR ||
			doFCUnPublishResult(event) == CMS_ERROR)
		{
			ret = CMS_ERROR;
		}
	}
	return ret;
}

int CRtmpProtocol::decodeCreateStream(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodeCreateStream] rtmp %s createStream #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	int ret = CMS_OK;
	string strEvent;
	amf0::Amf0Type type = amf0::amf0Block5Value(block,1,strEvent);
	if (type == amf0::AMF0_TYPE_NUMERIC)
	{
		int event = atoi(strEvent.c_str());
		ret = doCreateStreamSucc(event);
	}
	return ret;
}

int CRtmpProtocol::decodePublish(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodePublish] rtmp %s publish #####",
		mremoteAddr.c_str(),getRtmpType().c_str());
	std::string strInstanceName;
	amf0::amf0Block5Value(block,3,strInstanceName);
	if (!strInstanceName.empty())
	{
		if (murl.at(murl.length()-1) != '/')
		{
			murl += "/";
		}
	}
	murl += strInstanceName;
	logs->info(">>> %s [CRtmpProtocol::decodePublish] rtmp %s publish url %s ",
		mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());

	if (!parseUrl(murl,mlinkUrl))
	{
		logs->error("*** %s [CRtmpProtocol::decodePublish] rtmp %s publish url %s parse fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
		return CMS_ERROR;
	}

	if (isLegalIp(mlinkUrl.host.c_str()))
	{
		//302 ��ַ
		murl = "rtmp://";
		murl += mlinkUrl.app;
		murl += "/";
		murl += mlinkUrl.instanceName;
		logs->debug(">>> %s [CRtmpProtocol::decodePublish] rtmp %s publish delete 302 ip url %s ",
			mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
		if (!parseUrl(murl,mlinkUrl))
		{
			logs->error("*** %s [CRtmpProtocol::decodePublish] rtmp %s publish delete 302 url %s parse fail ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
			return CMS_ERROR;
		}
	}	
	msuper->setUrl(murl);
	if (misFMLEPublish)
	{
		doOnFCPublish();
	}
	mulNodelayEndTime = getTickCount();
	return doPublishSucc();
}

int CRtmpProtocol::decodePlay(amf0::Amf0Block *block)
{
	logs->info("#####%s [CRtmpProtocol::decodePlay] rtmp %s recv play command ",
		mremoteAddr.c_str(),getRtmpType().c_str());
	std::string strInstanceName;
	amf0::amf0Block5Value(block,3,strInstanceName);
	if (!strInstanceName.empty())
	{
		if (murl.at(murl.length()-1) != '/')
		{
			murl += "/";
		}
	}
	murl += strInstanceName;
	if (murl.find("/hash/") != string::npos)
	{
		murl = getBase64Decode(strInstanceName);
	}
	logs->info(">>> %s [CRtmpProtocol::decodePlay] rtmp %s play url %s ",
		mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());

	if (!parseUrl(murl,mlinkUrl))
	{
		logs->error("*** %s [CRtmpProtocol::decodePlay] rtmp %s publish url %s parse fail ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
		return CMS_ERROR;
	}

	if (isLegalIp(mlinkUrl.host.c_str()))
	{
		//302 ��ַ
		murl = "rtmp://";
		murl += mlinkUrl.app;
		murl += "/";
		murl += mlinkUrl.instanceName;
		logs->debug(">>> %s [CRtmpProtocol::decodePlay] rtmp %s publish delete 302 ip url %s ",
			mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
		if (!parseUrl(murl,mlinkUrl))
		{
			logs->error("*** %s [CRtmpProtocol::decodePlay] rtmp %s publish delete 302 url %s parse fail ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),murl.c_str());
			return CMS_ERROR;
		}
	}	
	msuper->setUrl(murl);
	moutStreamID = 1;
	if (doStreamBegin() == CMS_ERROR ||
		doPlayReset(strInstanceName) == CMS_ERROR ||
		doPlayStart(strInstanceName) == CMS_ERROR ||
		doSampleAccess() == CMS_ERROR || 
		doStreamDataStart() == CMS_ERROR ||
		doPlayPublishNotify(strInstanceName) == CMS_ERROR)
	{
		return CMS_ERROR;
	}
	msuper->tryCreateTask();
	int ret = msuper->doTransmission();
	if (ret < 0)
	{
		return CMS_ERROR;
	}
	mulNodelayEndTime = getTickCount();
	if (ret == 0 || ret == 2)
	{
		if (ret == 2)
		{
			//create task
		}
		doWriteTimeout();
	}
	return CMS_OK;
}

int CRtmpProtocol::sendMetaData(Slice *s)
{
	int iBodyLen = htonl(s->miDataLen);
	char fmt = CHUNK_STREAM_ID_USER_CONTROL;
	char timestamp[3] = {0};
	if (sendPacket(fmt,timestamp,NULL,((char*)&iBodyLen)+1,MESSAGE_TYPE_INVOKE,
		(const char*)&moutStreamID,s->mData,s->miDataLen) == false)
	{
		logs->error("*** %s [CRtmpProtocol::sendMetaData] rtmp %s send sendMetaData fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CRtmpProtocol::sendVideoOrAudio(Slice *s,uint32 uiTimestamp)
{	
	/*if (s->miDataType == DATA_TYPE_VIDEO)
	{
		logs->debug("###%s [CRtmpProtocol::sendVideoOrAudio] rtmp %s video size=%d,timestamp=%u",
			mremoteAddr.c_str(),getRtmpType().c_str(),s->miDataLen,s->muiTimestamp);
	}
	else
	{
		logs->debug("###%s [CRtmpProtocol::sendVideoOrAudio] rtmp %s audio size=%d,timestamp=%u",
			mremoteAddr.c_str(),getRtmpType().c_str(),s->miDataLen,s->muiTimestamp);
	}*/
	shouldCloseNodelay();
	int iBodyLen = htonl(s->miDataLen);
	char fmt = CHUNK_STREAM_VIDEO_AUDIO;
	char timestamp[3] = {0};
	char exTimstamp[4] = {0};
	//char msgLen[3] = {0};
	//bigPutInt24(msgLen,s->miDataLen);
	bool isHaveExTimestamp = false;
	if (uiTimestamp >= TIMESTAMP_EXTENDED)
	{
		isHaveExTimestamp = true;
		unsigned int ts = TIMESTAMP_EXTENDED;
		char *p = timestamp;
		char *v = (char*)&ts;
		*p++ = v[2];
		*p++ = v[1];
		*p++ = v[0];

		p = exTimstamp;
		v = (char*)&uiTimestamp;
		*p++ = v[3];
		*p++ = v[2];
		*p++ = v[1];
		*p++ = v[0];
	}
	else
	{
		char *p = timestamp;
		char *v = (char*)&uiTimestamp;
		*p++ = v[2];
		*p++ = v[1];
		*p++ = v[0];
	}

	char msgType = MESSAGE_TYPE_VIDEO;
	if (s->miDataType == DATA_TYPE_AUDIO || s->miDataType == DATA_TYPE_FIRST_AUDIO)
	{
		msgType = MESSAGE_TYPE_AUDIO;
	}
	if (sendPacket(fmt,timestamp,isHaveExTimestamp?exTimstamp:NULL,((char*)&iBodyLen)+1,msgType,
		(const char*)&moutStreamID,s->mData,s->miDataLen) == false)
	{
		logs->error("*** %s [CRtmpProtocol::sendVideoOrAudio] rtmp %s send sendVideoOrAudio fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}
	return CMS_OK;
}

bool CRtmpProtocol::sendPacket(char fmt,const char *timestamp,char *extentimestamp,const char *bodyLen,
									   char type,const char *streamId,const char *pData,int len)
{	
	int iHandleLen = 0;
	int ret = 0;
	mwrBuff->writeByte(fmt);
	mwrBuff->writeBytes(timestamp,3);
	mwrBuff->writeBytes(bodyLen,3);
	mwrBuff->writeByte(type);
	mwrBuff->writeBytes(streamId,4);
	if (extentimestamp)
	{
		mwrBuff->writeBytes(extentimestamp,4);
	}
	int iMaxLen = len > mwriteChunkSize ? mwriteChunkSize : len;
	ret = mwrBuff->writeBytes(pData,iMaxLen);
	if (ret == CMS_ERROR)
	{
		logs->error("*** %s [CRtmpProtocol::sendPacket] 1 rtmp %s send sendPacket fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return false;
	}
	iHandleLen += iMaxLen;
	char chunkID = (char)(fmt & 0x3F);
	char sameFmt = (HEADER_FORMAT_CONTINUATION << 6) | chunkID;
	while (len > iHandleLen)//��Ҫ�ֿ鴫��
	{
		mwrBuff->writeByte(sameFmt);
		if (extentimestamp)
		{
			mwrBuff->writeBytes(extentimestamp,4);
		}
		int iMaxLen = (len-iHandleLen) > mwriteChunkSize ? mwriteChunkSize : (len-iHandleLen);
		ret = mwrBuff->writeBytes(pData+iHandleLen,iMaxLen);
		iHandleLen += iMaxLen;

		if (ret == CMS_ERROR)
		{
			logs->error("*** %s [CRtmpProtocol::sendPacket] 2 rtmp %s send sendPacket fail,errno=%d,strerrno=%s ***",
				mremoteAddr.c_str(),getRtmpType().c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
			return false;
		}
	}	
	if (mwrBuff->size() > 0)
	{
		//�첽
		//ev_io_start(msuper->evLoop(), msuper->evWriteIO());
		msuper->evWriteIO();
	}
	else if (mrtmpType == RtmpClient2Publish || mrtmpType == RtmpServerBPlay)
	{
		//ev_io_stop(msuper->evLoop(), msuper->evWriteIO());
		//û�����ݿɶ�
		doWriteTimeout();
	}
	return true;
}

void CRtmpProtocol::syncIO()
{
	if (mwrBuff->size() > 0)
	{
		//�첽
		//ev_io_start(msuper->evLoop(), msuper->evWriteIO());
		msuper->evWriteIO();
	}
	else if (mrtmpType == RtmpClient2Publish || mrtmpType == RtmpServerBPlay)
	{
		//ev_io_stop(msuper->evLoop(), msuper->evWriteIO());
		//û�����ݿɶ�
		doWriteTimeout();
	}
}

int CRtmpProtocol::writeBuffSize()
{
	return mwrBuff->size();
}

std::string CRtmpProtocol::remoteAddr()
{
	return mremoteAddr;
}

std::string CRtmpProtocol::getUrl()
{
	return murl;
}

void CRtmpProtocol::doWriteTimeout()
{
	if (mcmsWriteTimeout == NULL)
	{
		mcmsWriteTimeout = mallcoCmsTimer();
		cms_timer_init(mcmsWriteTimeout,mrw->fd(),wait2WriteEV);
		assert(mcmsWriteTimeOutDo == 0);
		mcmsWriteTimeOutDo++;
		cms_timer_start(mcmsWriteTimeout);
	}
	else
	{
		if (mcmsWriteTimeOutDo == 0)
		{
			mcmsWriteTimeOutDo++;
			cms_timer_start(mcmsWriteTimeout);			
		}		
	}
}

void CRtmpProtocol::doReadTimeout()
{
	if (mcmsReadTimeout == NULL)
	{
		mcmsReadTimeout = mallcoCmsTimer();
		cms_timer_init(mcmsReadTimeout,mrw->fd(),wait2ReadEV,false);
		assert(mcmsReadTimeOutDo == 0);
		mcmsReadTimeOutDo++;
		cms_timer_start(mcmsReadTimeout,false);
	}
	else
	{
		if (mcmsReadTimeOutDo == 0)
		{
			mcmsReadTimeOutDo++;
			cms_timer_start(mcmsReadTimeout,false);			
		}		
	}
}

void CRtmpProtocol::shouldCloseNodelay()
{
	if (!misCloseNodelay && mulNodelayEndTime > 0)
	{
		if (getTickCount() - mulNodelayEndTime >= 1000*3)
		{
			logs->debug("#####%s [CRtmpProtocol::shouldCloseNodelay] rtmp %s close conn nodelay ",
				mremoteAddr.c_str(),getRtmpType().c_str());
			mrw->setNodelay(0);
			misCloseNodelay = true;
		}
	}
}

string CRtmpProtocol::getRtmpType()
{
	if (mrtmpType == RtmpClient2Play)
	{
		return "client 2 play";
	}
	if (mrtmpType == RtmpClient2Publish)
	{
		return "client 2 publish";
	}
	if (mrtmpType == RtmpServerBPlay)
	{
		return "server be play";
	}
	if (mrtmpType == RtmpServerBPublish)
	{
		return "server be publish";
	}
	if (mrtmpType == RtmpServerBPlayOrPublish)
	{
		return "server be play or publish";
	}
	return "unknow server tyep";
}

cms_timer *CRtmpProtocol::cmsTimer2Write()
{
	return mcmsWriteTimeout;
}

cms_timer *CRtmpProtocol::cmsTimer2Read()
{
	return mcmsReadTimeout;
}
