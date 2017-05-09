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
#include <protocol/cms_flv_pump.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <protocol/cms_flv.h>

CFlvPump::CFlvPump(CStreamInfo *super,HASH &hash,uint32 &hashIdx,std::string remoteAddr,std::string modeName,std::string url)
{
	msuper = super;
	mhash = hash;
	mhashIdx = hashIdx;
	miWidth = 0;
	miHeight = 0;
	miMediaRate = 0;
	miVideoFrameRate = 0;
	miVideoRate = 0;
	miAudioFrameRate = 0;
	miAudioRate = 0;
	miAudioSamplerate = 0;
	mvideoType = 0xFF;
	maudioType = 0xFF;
	//
	mremoteAddr = remoteAddr;
	mmodeName = modeName;
	murl = url;
	mjitter = new CJitter();
	mjitter->init(remoteAddr,modeName,url);
	mjitter->setOpenJitter(true);
	mcreateTT = getTimeUnix();

	mllIdx = 1;
	misPublish = false;
	misPushFlv = false;
}

CFlvPump::~CFlvPump()
{
	delete mjitter;
}

int	CFlvPump::decodeMetaData(char *data,int len,bool &isChangeMediaInfo)
{
	amf0::Amf0Block *block = NULL;
	block = amf0::amf0Parse(data,len);	
	std::string strRtmpContent = amf0::amf0BlockDump(block);
	logs->info("%s [ChttpClient::decodeMetaData] http %s received metaData: %s",
		mremoteAddr.c_str(),murl.c_str(),strRtmpContent.c_str());
	string strRtmpData = amf0::amf0Block2String(block);

	string rate;
	if (amf0::amf0Block5Value(block,"videodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate = atoi(rate.c_str()); //��Ƶ����
	}
	if (amf0::amf0Block5Value(block,"audiodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate += atoi(rate.c_str()); //��Ƶ����
	}
	string value;
	if (amf0::amf0Block5Value(block,"width",value) != amf0::AMF0_TYPE_NONE)
	{
		miWidth = atoi(value.c_str()); //��Ƶ�߶�
	}
	if (amf0::amf0Block5Value(block,"height",value) != amf0::AMF0_TYPE_NONE)
	{
		miHeight += atoi(value.c_str()); //��Ƶ���
	}
	//������
	if (amf0::amf0Block5Value(block,"framerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miVideoFrameRate = atoi(rate.c_str()); 
	}
	if (amf0::amf0Block5Value(block,"audiosamplerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miAudioSamplerate = atoi(rate.c_str()); 
		miAudioFrameRate = ::getAudioFrameRate(miAudioSamplerate);
	}

	mjitter->setVideoFrameRate(miVideoFrameRate);
	mjitter->setAudioFrameRate(miAudioFrameRate);

	isChangeMediaInfo = true;
	logs->info("%s [CFlvPump::decodeMetaData] http %s stream media rate=%d,width=%d,height=%d,video framerate=%d,audio samplerate=%d,audio framerate=%d",
		mremoteAddr.c_str(),murl.c_str(),strRtmpContent.c_str(),miMediaRate,miWidth,miHeight,miVideoFrameRate,miAudioSamplerate,miAudioFrameRate);

	Slice *s = newSlice();
	if (isChangeMediaInfo)
	{
		copy2Slice(s);
	}
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mhash;
	s->misMetaData = true;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	CFlvPool::instance()->push(mhashIdx,s);
	misPushFlv = true;
	return 1;
}

int CFlvPump::decodeVideo(char *data,int len,uint32 timestamp,bool &isChangeMediaInfo)
{
	if (len <= 0)
	{
		return 0;
	}

	//jitter
	int64 tn = getTimeUnix();
	int64 tk = tn - mcreateTT;
	if (mjitter->countVideoAudioFrame(true, tk, timestamp))
	{

	}
	//�����ľ���ʱ���ֻ��Դ�ͺ�����Ч
	if (mjitter->isOpenJitter())
	{
		timestamp = mjitter->judgeNeedJitter(true, timestamp);
	}
	//jitter end

	byte vType = byte(data[0] & 0x0F);
	if (mvideoType == 0xFF || mvideoType != vType)
	{
		if (mvideoType == 0xFF)
		{
			logs->info("%s [CFlvPump::decodeVideo] http %s first video type %s",
				mremoteAddr.c_str(),murl.c_str(),::getVideoType(vType).c_str());
		}
		else
		{
			logs->info("%s [CFlvPump::decodeVideo] http %s first video type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),::getVideoType(mvideoType).c_str(),::getVideoType(vType).c_str());
		}
		mvideoType = vType;
		isChangeMediaInfo = true;
	}
	bool isKeyFrame = false;
	FlvPoolDataType dataType = DATA_TYPE_VIDEO;
	if (vType == 0x02)
	{
		if (len <= 1)
		{
			return 0; //��֡
		}
	}
	else if (vType == 0x07)
	{
		if (len <= 5)
		{
			return 0; //��֡
		}
		if (data[0] == 0x17)
		{
			isKeyFrame = true;
			if (data[1] == 0x00)
			{
				dataType = DATA_TYPE_FIRST_VIDEO;
				miVideoFrameRate = 30;
				isChangeMediaInfo = true;
			}
			else if (data[1] == 0x01)
			{

			}
		}
	}
	Slice *s = newSlice();
	if (isChangeMediaInfo)
	{
		copy2Slice(s);
	}
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mhash;
	s->miDataType = dataType;
	s->misKeyFrame = isKeyFrame;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	if (dataType != DATA_TYPE_FIRST_VIDEO)
	{		
		mllIdx++;
		s->muiTimestamp = timestamp;
	}
	CFlvPool::instance()->push(mhashIdx,s);
	misPushFlv = true;
	return 1;
}

int CFlvPump::decodeAudio(char *data,int len,uint32 timestamp,bool &isChangeMediaInfo)
{
	if (len <= 0)
	{
		return 0;//��֡
	}

	//jitter
	int64 tn = getTimeUnix();
	int64 tk = tn - mcreateTT;
	if (mjitter->countVideoAudioFrame(false, tk, timestamp))
	{

	}
	//�����ľ���ʱ���ֻ��Դ�ͺ�����Ч
	if (mjitter->isOpenJitter())
	{
		timestamp = mjitter->judgeNeedJitter(false, timestamp);
	}
	//jitter end

	byte aType = byte(data[0]>>4 & 0x0F);
	if (maudioType == 0xFF || maudioType != aType)
	{
		if (maudioType == 0xFF)
		{
			logs->info("%s [CFlvPump::decodeAudio] http %s first audio type %s",
				mremoteAddr.c_str(),murl.c_str(),::getAudioType(aType).c_str());
		}
		else
		{
			logs->info("%s [CFlvPump::decodeAudio] http %s first audio type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),::getAudioType(maudioType).c_str(),::getAudioType(aType).c_str());
		}
		maudioType = aType;
		isChangeMediaInfo = true;
	}
	FlvPoolDataType dataType = DATA_TYPE_AUDIO;
	if (aType == 0x02 ||
		aType == 0x04 ||
		aType == 0x05 ||
		aType == 0x06 ){
			if (len <= 1)
			{
				return 0;//��֡
			}
	} 
	else if (aType >= 0x0A )
	{
		if (len >= 2 &&
			data[1] == 0x00)
		{
			if (len <= 2)
			{
				return 0;//��֡
			}
			dataType = DATA_TYPE_FIRST_AUDIO;
			if (len >= 4)
			{
				miAudioSamplerate = ::getAudioSampleRates(data);
				miAudioFrameRate = ::getAudioFrameRate(miAudioSamplerate);
				logs->info("%s [CFlvPump::decodeAudio] http %s audio sampleRate=%d,audio frame rate=%d",
					mremoteAddr.c_str(),murl.c_str(),miAudioSamplerate,miAudioFrameRate);
			}
		}
	}
	Slice *s = newSlice();
	if (isChangeMediaInfo)
	{
		copy2Slice(s);
	}
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mhash;
	s->miDataType = dataType;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	if (dataType != DATA_TYPE_FIRST_AUDIO)
	{
		mllIdx++;
		s->muiTimestamp = timestamp;
	}
	CFlvPool::instance()->push(mhashIdx,s);
	misPushFlv = true;
	return 1;
}

int CFlvPump::getWidth()
{
	return miWidth;
}

int CFlvPump::getHeight()
{
	return miHeight;
}

int CFlvPump::getMediaRate()
{
	return miMediaRate;
}

int CFlvPump::getVideoFrameRate()
{
	return miVideoFrameRate;
}

int CFlvPump::getVideoRate()
{
	return miVideoRate;
}

int CFlvPump::getAudioFrameRate()
{
	return miAudioFrameRate;
}

int CFlvPump::getAudioRate()
{
	return miAudioRate;
}

int CFlvPump::getAudioSampleRate()
{
	return miAudioSamplerate;
}

byte CFlvPump::getVideoType()
{
	return mvideoType;
}

byte CFlvPump::getAudioType()
{
	return maudioType;
}

void CFlvPump::stop()
{
	Slice *s = newSlice();
	s->mhHash = mhash;
	s->misPushTask = misPublish;
	s->misRemove = true;
	CFlvPool::instance()->push(mhashIdx,s);
}

void CFlvPump::setPublish()
{
	misPublish = true;
}

void CFlvPump::copy2Slice(Slice *s)
{
	s->misHaveMediaInfo = true;
	s->miVideoFrameRate = miVideoFrameRate;
	s->miVideoRate = miVideoRate;
	s->miAudioFrameRate = miAudioFrameRate;
	s->miAudioRate = miAudioRate;		
	s->miFirstPlaySkipMilSecond = msuper->firstPlaySkipMilSecond();
	s->misResetStreamTimestamp = msuper->isResetStreamTimestamp();
	s->mstrUrl = murl;
	s->miMediaRate = miMediaRate;
	s->miVideoRate = miVideoRate;
	s->miAudioRate = miAudioRate;
	s->miVideoFrameRate = miVideoFrameRate;
	s->miAudioFrameRate = miAudioFrameRate;
	s->misNoTimeout = msuper->isNoTimeout();
	s->mstrVideoType = ::getVideoType(mvideoType);
	s->mstrAudioType = ::getAudioType(maudioType);
	s->miLiveStreamTimeout = msuper->liveStreamTimeout();
	s->miNoHashTimeout = msuper->noHashTimeout();
	s->mstrRemoteIP = msuper->getRemoteIP();
	s->mstrHost = msuper->getHost();
	s->misRealTimeStream = msuper->isRealTimeStream();
	s->mllCacheTT = msuper->cacheTT();

	msuper->makeOneTask();
}