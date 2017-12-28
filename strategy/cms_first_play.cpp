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
#include <strategy/cms_first_play.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>

#define FIRST_DROP_MEDIA_RATE	400

CFirstPlay::CFirstPlay()
{
	misInit = false;
	mfirstPlaySkipMilSecond = -1;
	mdistanceKeyFrame = -1;
	mdropSliceNum = 0;
	mhaveDropSliceNum = 0;
	misSetFirstFrame = false;
	mvideoFrameRate = 0;
	maudioFrameRate = 0;
}

CFirstPlay::~CFirstPlay()
{

}

bool CFirstPlay::isInit()
{
	return misInit;
}

void CFirstPlay::init(HASH &hash,uint32 &hashIdx,std::string remoteAddr,std::string modeName,std::string url)
{
	misInit = true;
	murl = url;
	mremoteAddr = remoteAddr;
	modeName = modeName;

	mhash = hash;
	mhashIdx = hashIdx;
}

bool CFirstPlay::checkfirstPlay()
{
	if (mfirstPlaySkipMilSecond == -1)
	{
		mfirstPlaySkipMilSecond = CFlvPool::instance()->getFirstPlaySkipMilSecond(mhashIdx,mhash);
		if (mfirstPlaySkipMilSecond == -1)
		{
			return false;
		}
	}
	if (mdistanceKeyFrame == -1)
	{
		mdistanceKeyFrame = (int32)CFlvPool::instance()->getDistanceKeyFrame(mhashIdx,mhash);
		if (mdistanceKeyFrame == 0)
		{
			//������ʱ��ᵼ���ʼ���û�����ʧ��
			mfirstPlaySkipMilSecond = 0;
		}
		beginTT = getTimeUnix();
		logs->debug(">>>%s %s first play task %s firstPlaySkipMilSecond=%d,distanceKeyFrame=%d",
			mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
			mfirstPlaySkipMilSecond,mdistanceKeyFrame);
	}	
	return true;
}

bool CFirstPlay::checkShouldDropFrameCount(int64 &transIdx,Slice *s)
{
	if (!(mfirstPlaySkipMilSecond > 0 && transIdx == -1 &&
			((s->mData[0] == VideoTypeAVCKey && CFlvPool::instance()->isH264(mhashIdx,mhash)) ||
			 (s->mData[0] == VideoTypeHEVCKey && CFlvPool::instance()->isH265(mhashIdx,mhash)))/* &&
		CFlvPool::instance()->getMediaRate(mhashIdx,mhash) > FIRST_DROP_MEDIA_RATE*/))
	{		
		return true;
	}
	if (getTimeUnix() - beginTT > 3)
	{
		//�����������
		mfirstPlaySkipMilSecond = 0;
		return true;
	}
	misSetFirstFrame = true;
	mvideoFrameRate = CFlvPool::instance()->getVideoFrameRate(mhashIdx,mhash);
	maudioFrameRate = CFlvPool::instance()->getAudioFrameRate(mhashIdx,mhash);
	if (mvideoFrameRate == -1 || mvideoFrameRate == 1)
	{
		return false;
	}
	if (mvideoFrameRate > 75)
	{
		mvideoFrameRate = 30;
	}
	logs->debug(">>>%s %s first play task %s video frame rate=%d,audio frame rate=%d",
		mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
		mvideoFrameRate,maudioFrameRate);
	if (mfirstPlaySkipMilSecond < mdistanceKeyFrame)
	{
		mdropSliceNum = (mvideoFrameRate+maudioFrameRate)*mfirstPlaySkipMilSecond/1000;		
		int64 minIdx = CFlvPool::instance()->getMinIdx(mhashIdx,mhash);
		int64 maxIdx = CFlvPool::instance()->getMaxIdx(mhashIdx,mhash);
		logs->debug(">>>1 %s %s first play task %s should drop slice num %d,minIdx=%lld,maxIdx=%lld,s->mllIndex=%lld",
			mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
			mdropSliceNum,minIdx,maxIdx,s->mllIndex);
		if (s->mllIndex-minIdx > (int64)mdropSliceNum)
		{
			transIdx = s->mllIndex-(int64)mdropSliceNum-1;
			if (transIdx-minIdx < 5)
			{
				transIdx += 15;
				mdropSliceNum -= 15;
			}
		}
		else
		{
			transIdx = minIdx+20-1;
			mdropSliceNum = (int32)(s->mllIndex-minIdx-20);
		}
		if (mdropSliceNum <= 0)
		{
			transIdx = -1;
			return true;
		}
	}
	else
	{
		int64 maxIdx = CFlvPool::instance()->getMaxIdx(mhashIdx,mhash);
		mdropSliceNum = (mvideoFrameRate+maudioFrameRate)*mfirstPlaySkipMilSecond/1000;
		mdropSliceNum -= 20;
		logs->debug(">>>2 %s %s first play task %s should drop slice num %d,maxIdx=%lld,s->mllIndex=%lld",
			mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
			mdropSliceNum,maxIdx,s->mllIndex);
		if (mdropSliceNum > 0)
		{
			transIdx = s->mllIndex+(int64)mdropSliceNum+10;
		}
		else
		{
			mdropSliceNum = 0;
			return true;
		}
	}
	return false;
}

bool CFirstPlay::needDropFrame(Slice *s)
{
	bool needDrop = false;
	if (s->miDataType == DATA_TYPE_VIDEO && mfirstPlaySkipMilSecond > 0 && misSetFirstFrame)
	{
		if (!((misSetFirstFrame &&
			mhaveDropSliceNum < mdropSliceNum-5) ||
			(misSetFirstFrame &&
			mhaveDropSliceNum >= mdropSliceNum-5 &&
			s->mData[0] != VideoTypeAVCKey &&
			s->mData[0] != VideoTypeHEVCKey))) 
		{
			//����ؼ�֡����gopС�ڶ�֡���ȣ�ֻ�����㶪֡�����ڶ�֡���ȣ����������ؼ�֡���ܽ���
			logs->debug(">>>3 %s %s first play task %s 11 should drop slice num %d,have drop %d",
				mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
				mdropSliceNum,mhaveDropSliceNum);
			misSetFirstFrame = false;
		}
		else
		{
			//����ؼ�֡����gop���ڶ�֡���ȣ������ؼ�֡�϶���֡����
			if (mdistanceKeyFrame > mfirstPlaySkipMilSecond &&
				(s->mData[0] == VideoTypeAVCKey || s->mData[0] == VideoTypeHEVCKey))
			{
				logs->debug(">>>4 %s %s first play task %s 22 should drop slice num %d,have drop %d",
					mremoteAddr.c_str(),modeName.c_str(),murl.c_str(),
					mdropSliceNum,mhaveDropSliceNum);
				misSetFirstFrame = false;
			}
			else
			{				
				needDrop = true;
			}
		}
	}
	if (misSetFirstFrame)
	{
		mhaveDropSliceNum++;
	}
	return needDrop;
}

