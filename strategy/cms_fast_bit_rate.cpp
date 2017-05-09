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
#include <strategy/cms_fast_bit_rate.h>
#include <flvPool/cms_flv_pool.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>

#define DropVideoCacheLen			5000
#define DropVideoStartTime			2000
#define DropVideoCheckInterval		200
#define DropVideoTimeDiffer			100
#define DropVideoClientBufLess		-800
#define DropVideoServerBufLess		2000
#define DropVideoLoseBufTime		5
#define DropVideoLoseBufMaxTime		10

int64 MathAbs(int64 value) 
{
	if (value < 0) 
	{
		return 0 - value;
	}
	return value;
}

CFastBitRate::CFastBitRate()
{
	misWaterMark = false;
	mwaterMarkOriHashIdx = 0;
	mhashIdx = 0;
	//�����л�
	mautoBitRateMode = AUTO_DROP_CHANGE_BITRATE_CLOSE;
	misChangeVideoBit = false;				//�Ƿ��л�������
	mbeginVideoChangeTime = 0;				//�ײ���֡
	mendVideoChangeTime = 0;	            //�ײ���֡
	mfirstLowBitRateTimeStamp = 0;			//�ײ������������ʱ����������л���������ʱ��ƥ����ѷ���λ��
	mhashLowBitRateIdx = 0;                 //��ǰ����������
	mmaxLowBitRateIdx = 0;                  //�����������
	mchangeLowBitRateTimes = 0;             //�����л�����
	mchangeBitRateTT = 0;
	mcreateBitRateTaskTT = 0;
	mautoBitRateFactor = 3;					//��̬������ϵ��
	mautoFrameFactor = 20;					//��̬��֡ϵ��
	//�����л���֤ʱ���ƽ��
	mlastVideoChangeBitRateTimestamp = 0;   //�л�����ǰ�����Ƶʱ���
	mlastVideoChangeBitRateRelativeTimestamp = 0;		//�л����ʺ�lastVideoChangeBitRateTimestamp �� �л����ʺ��һ֡��Ƶ֡ʱ�������ֵ
	mneedChangeVideoTimestamp = false;                  //�л����ʺ���Ҫ����ʱ�����־
	mlastAudioChangeBitRateTimestamp = 0;				//�л�����ǰ�����Ƶʱ���
	mlastAudioChangeBitRateRelativeTimestamp = 0;		//�л����ʺ�lastVideoChangeBitRateTimestamp �� �л����ʺ��һ֡��Ƶ֡ʱ�������ֵ
	mneedChangeAudioTimestamp = false;					//�л����ʺ���Ҫ����ʱ�����־
	//��֡����
	mno1AudioTimestamp = 0;					//��֡��Ƶʱ���
	mno1VideoTimestamp = 0;					//��֡��Ƶʱ���
	mconnectTimestamp = 0;					//����ʱ���
	mtransCodeNeedDropVideo = false;		//�Ƿ���Ҫ��֡
	mtransCodeNoNeedDropVideo = false;		//�Ƿ���Ҫ��֡
	mlastDiffer = 0;						//�ϴ�ʱ�����
	mloseBufferTimes = 0;					//Ԥ�⿨�ٴ���
	mloseBufferInterval = getTickCount();	//��¼ͳ�Ƽ��ʱ��
	mlastVideoTimestamp = 0;
	mbeginDropVideoTimestamp = 0;
	//��֡�����л�
	mhistoryDropTotal = 0;
	mhistoryDropNum = 0;
	mhistoryDropTime = 0;
	mhistoryDropTT = 0;
	misInit = false;
}

CFastBitRate::~CFastBitRate()
{

}

bool CFastBitRate::isInit()
{
	return misInit;
}

void CFastBitRate::init(std::string remoteAddr, std::string modeName, std::string url ,bool isWaterMark ,
	uint32 waterMarkOriHashIdx,uint32 hashIdx, HASH &waterMarkOriHash, HASH &hash) 
{
	misInit = true;
	murl = url;
	mremoteAddr = remoteAddr;
	modeName = modeName;

	mwaterMarkOriHashIdx = waterMarkOriHashIdx;
	mwaterMarkOriHash = waterMarkOriHash;
	misWaterMark = isWaterMark;

	mhashIdx = hashIdx;
	mhash = hash;
}

void CFastBitRate::setFirstLowBitRateTimeStamp(uint32 timestamp) 
{
	mfirstLowBitRateTimeStamp = timestamp;
}

void CFastBitRate::changeBitRateLastTimestamp(int32 dateType,uint32 timestamp) 
{
	if (dateType == DATA_TYPE_VIDEO)
	{
		if (mbeginVideoChangeTime == 0) 
		{
			mbeginVideoChangeTime = timestamp;
		}
		mendVideoChangeTime = timestamp;
		mlastVideoChangeBitRateTimestamp = timestamp;
		if (mlastAudioChangeBitRateTimestamp > mlastVideoChangeBitRateTimestamp)
		{
			mlastVideoChangeBitRateTimestamp = mlastAudioChangeBitRateTimestamp;
		}
	} 
	else if (dateType == DATA_TYPE_AUDIO) 
	{
		mlastAudioChangeBitRateTimestamp = timestamp;
		if (mlastVideoChangeBitRateTimestamp > mlastAudioChangeBitRateTimestamp)
		{
			mlastAudioChangeBitRateTimestamp = mlastVideoChangeBitRateTimestamp;
		}
	}
}

uint32 CFastBitRate::changeBitRateSetTimestamp(int32 dateType,uint32 sendTimestamp) 
{
	if (dateType == DATA_TYPE_VIDEO && mneedChangeVideoTimestamp)
	{
		mneedChangeVideoTimestamp = false;
		mlastVideoChangeBitRateRelativeTimestamp = int32(mlastVideoChangeBitRateTimestamp) - int32(sendTimestamp);
	} 
	else if (dateType == DATA_TYPE_AUDIO && mneedChangeAudioTimestamp)
	{
		mneedChangeAudioTimestamp = false;
		mlastAudioChangeBitRateRelativeTimestamp = int32(mlastAudioChangeBitRateTimestamp) - int32(sendTimestamp);
	}
	if (dateType == DATA_TYPE_VIDEO)
	{
		if (mlastVideoChangeBitRateRelativeTimestamp > 0)
		{
			sendTimestamp = sendTimestamp + uint32(mlastVideoChangeBitRateRelativeTimestamp) + 50;
		}
		else if (mlastVideoChangeBitRateRelativeTimestamp < 0)
		{
			sendTimestamp = sendTimestamp - uint32(mlastVideoChangeBitRateRelativeTimestamp*-1) + 50;
		}
		mlastVideoChangeBitRateTimestamp = sendTimestamp;
		if (mlastAudioChangeBitRateTimestamp > mlastVideoChangeBitRateTimestamp)
		{
			mlastVideoChangeBitRateTimestamp = mlastAudioChangeBitRateTimestamp;
		}
		mlastAudioChangeBitRateTimestamp = mlastVideoChangeBitRateTimestamp;
	} 
	else if (dateType == DATA_TYPE_AUDIO)
	{
		if (mlastAudioChangeBitRateRelativeTimestamp > 0)
		{
			sendTimestamp = sendTimestamp + uint32(mlastAudioChangeBitRateRelativeTimestamp) + 50;
		} 
		else if (mlastAudioChangeBitRateRelativeTimestamp < 0)
		{
			sendTimestamp = sendTimestamp - uint32(mlastAudioChangeBitRateRelativeTimestamp*-1) + 50;
		}
		mlastAudioChangeBitRateTimestamp = sendTimestamp;
		if (mlastVideoChangeBitRateTimestamp > mlastAudioChangeBitRateTimestamp)
		{
			mlastAudioChangeBitRateTimestamp = mlastVideoChangeBitRateTimestamp;
		}
		mlastVideoChangeBitRateTimestamp = mlastAudioChangeBitRateTimestamp;
	}
	return sendTimestamp;
}

void CFastBitRate::setChangeBitRate() 
{
	std::string suffixName = "";
	std::string transCodeSuffix = "";
	if (!misWaterMark) 
	{
		mautoBitRateMode = CFlvPool::instance()->readBitRateMode(mhashIdx, mhash);
		suffixName = CFlvPool::instance()->readChangeBitRateSuffix(mhashIdx, mhash);
		transCodeSuffix = CFlvPool::instance()->readCodeSuffix(mhashIdx, mhash);
	} else 
	{
		mautoBitRateMode = CFlvPool::instance()->readBitRateMode(mwaterMarkOriHashIdx, mwaterMarkOriHash);
		suffixName = CFlvPool::instance()->readChangeBitRateSuffix(mwaterMarkOriHashIdx, mwaterMarkOriHash);
		transCodeSuffix = CFlvPool::instance()->readCodeSuffix(mwaterMarkOriHashIdx, mwaterMarkOriHash);
	}

	logs->debug("fast bit rate %s mautoBitRateMode=%d,suffixName=%s,transCodeSuffix=%s.",
		murl.c_str(), mautoBitRateMode, suffixName.c_str(), transCodeSuffix.c_str());
	if ((mautoBitRateMode == AUTO_CHANGE_BITRATE_OPEN ||
		mautoBitRateMode == AUTO_DROP_CHANGE_BITRATE_OPEN) &&
		!suffixName.empty() && transCodeSuffix != suffixName)
	{
		mhashLowBitRate.clear();
		mhashIdxLowBitRate.clear();

		std::string oriUrl = murl;
		std::size_t i = oriUrl.find("?");
		if (i != std::string::npos)
		{
			oriUrl = oriUrl.substr(0,i);
		}
		i = oriUrl.find(".flv");
		if (i != std::string::npos)
		{
			oriUrl = oriUrl.substr(0,i);
			if (!transCodeSuffix.empty())
			{
				i = oriUrl.rfind(transCodeSuffix);
				if (i != std::string::npos)
				{
					if (i == oriUrl.length()-transCodeSuffix.length()) 
					{
						//��ҪУ�飬��������ַ�����ȫƥ��ſ���
						oriUrl = oriUrl.substr(0,i);
					}
				}
			}
			oriUrl += suffixName;
			oriUrl += ".flv";
		} 
		else 
		{
			if (!transCodeSuffix.empty()) 
			{
				i = oriUrl.rfind(transCodeSuffix);
				if (i != std::string::npos)
				{
					if (i == oriUrl.length()-transCodeSuffix.length())
					{
						//��ҪУ�飬��������ַ�����ȫƥ��ſ���
						oriUrl = oriUrl.substr(0,i);
					}
				}
			}
		}
		mhashLowBitRate.push_back(mhash);
		mhashIdxLowBitRate.push_back(mhashIdx);

		oriUrl += suffixName;
		mlowBitRateUrl = oriUrl;
		std::string hashUrlLow = readHashUrl(oriUrl);
		logs->debug("##### fast bit rate %s low bit rate url %s,low bit rate hash url %s #####",
			murl.c_str(),mlowBitRateUrl.c_str(), hashUrlLow.c_str());
		HASH hashLow = makeHash(hashUrlLow.c_str(),hashUrlLow.length());
		mhashLowBitRate.push_back(hashLow);
		uint32 hashIdx = CFlvPool::instance()->hashIdx(hashLow);
		mhashIdxLowBitRate.push_back(hashIdx);
		mlowBitRateHash = hashLow;

		mmaxLowBitRateIdx = 1;
		mhashLowBitRateIdx = 0;
	} 
	else 
	{
		if (mautoBitRateMode == AUTO_DROP_CHANGE_BITRATE_OPEN)
		{
			mautoBitRateMode = AUTO_DROP_BITRATE_OPEN;
		} 
		else if (mautoBitRateMode == AUTO_CHANGE_BITRATE_OPEN)
		{
			mautoBitRateMode = AUTO_DROP_CHANGE_BITRATE_CLOSE;
		} 
		else if (mautoBitRateMode == AUTO_DROP_BITRATE_OPEN) 
		{
			//mautoBitRateMode = AUTO_DROP_BITRATE_OPEN;
		} 
		else 
		{
			mautoBitRateMode = AUTO_DROP_CHANGE_BITRATE_CLOSE;
		}
	}
}

void CFastBitRate::resetDropFlag() 
{
	while (!mhistoryDropList.empty())
	{
		mhistoryDropList.pop();
	}
	mhistoryDropTotal = 0;
	mhistoryDropNum = 0;
}

int CFastBitRate::dropFramePer(int64 te,int32 sliceFrameRate) 
{
	//��ⶪ֡��
	int dropPer = 0;
	if (te-mhistoryDropTT > 3000)
	{
		mhistoryDropTT = te;
		mhistoryDropTotal += mhistoryDropNum;
		mhistoryDropList.push(mhistoryDropNum);
		if (mhistoryDropList.size() > 40)
		{
			int num = mhistoryDropList.front();
			mhistoryDropTotal -= num;
			mhistoryDropList.pop();
		}
		mhistoryDropNum = 0;
		if (sliceFrameRate > 0)
		{
			dropPer = mhistoryDropTotal * 100 / int(sliceFrameRate*120);
			if (dropPer >= 10)
			{
				logs->debug("##### fast bit rate %s need change to lowbit #####",murl.c_str());
			}
		}
	}
	//��ⶪ֡�ʽ���
	return dropPer;
}

void CFastBitRate::dropOneFrame() 
{
	mhistoryDropNum++;
}

bool CFastBitRate::changeRateBit(uint32 hashIdxOld ,HASH &hashOld ,bool isReset ,
	uint32 timestamp,std::string referUrl,uint32 &hashIdxRead ,HASH &hashRead,int64 &transIdx,int32 &mode) 
{
	bool isSucc = false;
	bool isRaise = false;
	hashRead = hashOld;
	hashIdxRead = hashIdxOld;
	if (!isReset)
	{
		if (mhashLowBitRateIdx >= mmaxLowBitRateIdx)
		{
			return false;
		}
		mhashLowBitRateIdx++;
		isRaise = true;
		mode = 0;
	} 
	else 
	{
		if (mhashLowBitRateIdx <= 0)
		{
			return false;
		}
		mhashLowBitRateIdx--;
		mode = 1;
	}
	do 
	{
		misChangeVideoBit = true;
		int64 tt = getTimeUnix();
		hashRead = mhashLowBitRate[mhashLowBitRateIdx];
		hashIdxRead = mhashIdxLowBitRate[mhashLowBitRateIdx];
		if (hashRead == mcreateBitRateHash && tt-mcreateBitRateTaskTT < 3)
		{
			//��ֹһֱ��������
			break;
		}
		if (hashRead == mlowBitRateHash && !CFlvPool::instance()->isExist(hashIdxRead, hashRead))
		{
			/*createTask(mlowBitRateUrl, hashRead, referUrl, CREATE_TASK_ACT_NORMAL)
			mcreateBitRateTaskTT = tt
			mcreateBitRateHash = hashRead*/
		}
		bool  res;
		int64 idx;
		res = CFlvPool::instance()->seekKeyFrame(hashIdxRead, hashRead, timestamp,idx); //flv.SeekSlice(r.hashIdxRead, r.hashRead, timestamp)
		if (!res)
		{
			logs->error("***fast bit rate %s changeRateBit not find SeekKeyFrame ***",murl.c_str());
			break;
		}

		//ͳ��ʱ�����λ
		mneedChangeAudioTimestamp = true;
		mneedChangeVideoTimestamp = true;
		transIdx = idx;
		isSucc = true;
	} while (0);
	if (!isSucc)
	{
		if (isRaise)
		{
			mhashLowBitRateIdx--;
		} 
		else 
		{
			mhashLowBitRateIdx++;
		}
		hashRead = hashOld;
		hashIdxRead = hashIdxOld;
		misChangeVideoBit = false;
	}
	return isSucc;
}

bool CFastBitRate::dropVideoFrame(int32 edgeCacheTT,int32 dataType,int32 sliceFrameRate,
								  int64 te,uint32 sendTimestamp,int sliceNum)
{
	bool isCaton = false;
	if (edgeCacheTT >= DropVideoCacheLen)
	{ 
		//��Ի���ʱ����Ƴ���10�����Ч
		if ((te - mconnectTimestamp) > DropVideoStartTime)
		{
			if (dataType == DATA_TYPE_AUDIO)
			{
				mlastVideoTimestamp = sendTimestamp;
				int64 differ = (te - mconnectTimestamp) - int64(sendTimestamp-mno1AudioTimestamp);
				int32 guestBufLen = 0;
				if (sliceFrameRate > 0)
				{
					guestBufLen = int32(sliceNum*1000) / sliceFrameRate; //ת���ɺ���
				}
				int64 compareDiffer = mlastDiffer - differ;
				if (te-mloseBufferInterval > DropVideoCheckInterval)
				{ 
					//200ms�ж�һ��
					if (MathAbs(compareDiffer) > DropVideoTimeDiffer)
					{
						if (compareDiffer < 0 && mloseBufferTimes < DropVideoLoseBufMaxTime)
						{
							mloseBufferTimes++;
						}
						if (compareDiffer > 0 && mloseBufferTimes > 0)
						{
							mloseBufferTimes--;
						}
					}
					//�ж��û�ʣ�໺������С��800ms
					if (differ >= DropVideoClientBufLess)
					{
						mloseBufferTimes = DropVideoLoseBufMaxTime;
					}
					//�жϷ���������������С��2000ms
					if (guestBufLen < DropVideoServerBufLess)
					{
						mloseBufferTimes = 0;
					}
					if (mtransCodeNeedDropVideo || mtransCodeNoNeedDropVideo)
					{
						logs->debug("one conn %s,%s %s #####AutoBitRateMode=%d,"\
							" $$$rtmp drop video frame differ=%lld,lastdiffer=%lld,guestBufLen=%d,"\
							"mloseBufferTimes=%d,int32(edgeCacheTT/2)=%d,te=%lld,r.connectTimestamp=%lld,sendTimestamp=%u,"\
							"mno1AudioTimestamp=%u,(te - r.connectTimestamp)=%lld,int64(sendTimestamp-mno1AudioTimestamp)=%lld",
							mremoteAddr.c_str(),
							modeName.c_str(),
							murl.c_str(),
							mautoBitRateMode,
							differ,
							mlastDiffer,
							guestBufLen,
							mloseBufferTimes,
							int32(edgeCacheTT/2),
							te,
							mconnectTimestamp,
							sendTimestamp,
							mno1AudioTimestamp,
							(te - mconnectTimestamp),
							int64(sendTimestamp-mno1AudioTimestamp));
					}
					mloseBufferInterval = te;
					mlastDiffer = differ;
					if (guestBufLen >= int32(edgeCacheTT/2))
					{
						if (mloseBufferTimes >= int32(mautoFrameFactor)) 
						{
							if (mautoBitRateMode == AUTO_CHANGE_BITRATE_OPEN)
							{
								if (!mtransCodeNoNeedDropVideo) 
								{
									isCaton = true;
								}
								mtransCodeNoNeedDropVideo = true;
							} 
							else 
							{
								if (!mtransCodeNeedDropVideo)
								{
									isCaton = true;
								}
								mbeginDropVideoTimestamp = sendTimestamp;
								mtransCodeNeedDropVideo = true;
							}
						}
					}
				}
			}
			if (dataType == DATA_TYPE_VIDEO)
			{
				int64 differ = (te - mconnectTimestamp) - int64(sendTimestamp-mno1VideoTimestamp);
				int32 guestBufLen = 0;
				if (sliceFrameRate > 0)
				{
					guestBufLen = int32(sliceNum*1000) / sliceFrameRate; //ת���ɺ���
				}
				int64 compareDiffer = mlastDiffer - differ;
				if (te-mloseBufferInterval > DropVideoCheckInterval)
				{
					if (MathAbs(compareDiffer) > DropVideoTimeDiffer)
					{
						if (compareDiffer < 0 && mloseBufferTimes < DropVideoLoseBufMaxTime)
						{
							mloseBufferTimes++;
						}
						if (compareDiffer > 0 && mloseBufferTimes > 0)
						{
							mloseBufferTimes--;
						}
					}
					//�ж��û�ʣ�໺������С��800ms
					if (differ >= DropVideoClientBufLess)
					{
						mloseBufferTimes = DropVideoLoseBufMaxTime;
					}
					//�жϷ���������������С��2000ms
					if (guestBufLen < DropVideoServerBufLess)
					{
						mloseBufferTimes = 0;
					}

					if (mtransCodeNeedDropVideo || mtransCodeNoNeedDropVideo) 
					{
						logs->debug("one conn %s,%s %s #####AutoBitRateMode=%d,"\
							" $$$rtmp drop video frame differ=%lld,lastdiffer=%lld,guestBufLen=%d,"\
							"mloseBufferTimes=%d,int32(edgeCacheTT/2)=%d,te=%lld,r.connectTimestamp=%lld,sendTimestamp=%u,"\
							"mno1VideoTimestamp=%u,(te - r.connectTimestamp)=%lld,int64(sendTimestamp-mno1VideoTimestamp)=%lld",
							mremoteAddr.c_str(),
							modeName.c_str(),
							murl.c_str(),
							mautoBitRateMode,
							differ,
							mlastDiffer,
							guestBufLen,
							mloseBufferTimes,
							int32(edgeCacheTT/2),
							te,
							mconnectTimestamp,
							sendTimestamp,
							mno1VideoTimestamp,
							(te - mconnectTimestamp),
							int64(sendTimestamp-mno1VideoTimestamp));
					} 
					mlastDiffer = differ;
					mloseBufferInterval = te;

					if (guestBufLen >= int32(edgeCacheTT/2))
					{
						if (mloseBufferTimes >= int32(mautoFrameFactor))
						{
							if (mautoBitRateMode == AUTO_CHANGE_BITRATE_OPEN)
							{
								if (!mtransCodeNoNeedDropVideo)
								{
									isCaton = true;
								}
								mtransCodeNoNeedDropVideo = true;
							} 
							else 
							{
								if (!mtransCodeNeedDropVideo)
								{
									isCaton = true;
								}
								mbeginDropVideoTimestamp = mlastVideoTimestamp;
								mtransCodeNeedDropVideo = true;
							}
						}
					}
				}
			}
		} 
		else 
		{
			if (dataType == DATA_TYPE_AUDIO)
			{
				mlastDiffer = (te - mconnectTimestamp) - int64(sendTimestamp-mno1AudioTimestamp);
			}
			if (dataType == DATA_TYPE_VIDEO)
			{
				mlastDiffer = (te - mconnectTimestamp) - int64(sendTimestamp-mno1VideoTimestamp);
			}
		}
	}
	return isCaton;
}

bool CFastBitRate::isDropEnoughTime(uint32 sendTimestamp)
{
	return sendTimestamp-mbeginDropVideoTimestamp > 2000;
}

void CFastBitRate::resetDropFrameFlags() 
{
	mno1AudioTimestamp = 0;
	mno1VideoTimestamp = 0;
	mconnectTimestamp = getTickCount();
	mtransCodeNeedDropVideo = false;
}

bool CFastBitRate::needResetFlags(int32 dateType,uint32 sendTimestamp)
{
	if ((dateType == DATA_TYPE_AUDIO && sendTimestamp < mno1AudioTimestamp) ||
		(dateType == DATA_TYPE_VIDEO && sendTimestamp < mno1VideoTimestamp))
	{
		return true;
	}
	return false;
}

void CFastBitRate::setNo1VideoAudioTimestamp(bool isVideo,uint32 sendTimestamp) 
{
	if (isVideo)
	{
		if (mno1VideoTimestamp == 0)
		{
			mno1VideoTimestamp = sendTimestamp;
			if (mconnectTimestamp == 0)
			{
				mconnectTimestamp = getTickCount();
			}
		}
	} 
	else 
	{
		if (mno1AudioTimestamp == 0)
		{
			mno1AudioTimestamp = sendTimestamp;
			if (mconnectTimestamp == 0)
			{
				mconnectTimestamp = getTickCount();
			}
		}
	}
}

bool CFastBitRate::isChangeBitRate()
{
	return misChangeVideoBit;
}

int	CFastBitRate::getAutoBitRateMode()
{
	return mautoBitRateMode;
}

int	CFastBitRate::getAutoBitRateFactor()
{
	return mautoBitRateFactor;
}

void CFastBitRate::setAutoBitRateFactor(int autoBitRateFactor)
{
	mautoBitRateFactor = autoBitRateFactor;
}

int	CFastBitRate::getAutoFrameFactor()
{
	return mautoFrameFactor;
}

void CFastBitRate::setAutoFrameFactor(int autoFrameFactor)
{
	mautoFrameFactor = autoFrameFactor;
}

bool CFastBitRate::getTransCodeNeedDropVideo()
{
	return mtransCodeNeedDropVideo;
}

void CFastBitRate::setTransCodeNeedDropVideo(bool is)
{
	mtransCodeNeedDropVideo = is;
}

bool CFastBitRate::getTransCodeNoNeedDropVideo()
{
	return mtransCodeNoNeedDropVideo;
}

void CFastBitRate::setTransCodeNoNeedDropVideo(bool is)
{
	mtransCodeNoNeedDropVideo = is;
}

int32 CFastBitRate::getLoseBufferTimes()
{
	return mloseBufferTimes;
}

bool CFastBitRate::isChangeVideoBit()
{
	return misChangeVideoBit;
}
