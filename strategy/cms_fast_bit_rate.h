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
#ifndef __CMS_FAST_BIT_RATE_H__
#define __CMS_FAST_BIT_RATE_H__
#include <common/cms_type.h>
#include <string>
#include <vector>
#include <queue>

#define DropVideoKeyFrameLen		4000
//��̬����
#define AUTO_DROP_CHANGE_BITRATE_CLOSE  0
#define AUTO_CHANGE_BITRATE_OPEN        1
#define AUTO_DROP_BITRATE_OPEN          2
#define AUTO_DROP_CHANGE_BITRATE_OPEN   3

class CFastBitRate
{
public:
	CFastBitRate();
	~CFastBitRate();
	bool	isInit();
	void	init(std::string remoteAddr, std::string modeName, std::string url ,bool isWaterMark ,
		uint32 waterMarkOriHashIdx,uint32 hashIdx, HASH &waterMarkOriHash, HASH &hash);
	void	setFirstLowBitRateTimeStamp(uint32 timestamp);
	void	changeBitRateLastTimestamp(int32 dateType,uint32 timestamp);
	uint32	changeBitRateSetTimestamp(int32 dateType,uint32 sendTimestamp);
	void	setChangeBitRate();
	void	resetDropFlag();
	int		dropFramePer(int64 te,int32 sliceFrameRate);
	void	dropOneFrame();
	bool	changeRateBit(uint32 hashIdxOld ,HASH &hashOld ,bool isReset ,
		uint32 timestamp,std::string referUrl,uint32 &hashIdxRead ,HASH &hashRead,int64 &transIdx,int32 &mode);
	bool	dropVideoFrame(int32 edgeCacheTT,int32 dataType,int32 sliceFrameRate,
		int64 te,uint32 sendTimestamp,int sliceNum);
	bool	isDropEnoughTime(uint32 sendTimestamp);
	void	resetDropFrameFlags();
	bool	needResetFlags(int32 dateType,uint32 sendTimestamp);
	void	setNo1VideoAudioTimestamp(bool isVideo,uint32 sendTimestamp);
	
	bool    isChangeBitRate();
	int		getAutoBitRateMode();
	int		getAutoBitRateFactor();
	void	setAutoBitRateFactor(int autoBitRateFactor);
	int		getAutoFrameFactor();
	void	setAutoFrameFactor(int autoFrameFactor);
	bool	getTransCodeNeedDropVideo();
	void	setTransCodeNeedDropVideo(bool is);
	bool	getTransCodeNoNeedDropVideo();
	void	setTransCodeNoNeedDropVideo(bool is);
	int32	getLoseBufferTimes();
	bool	isChangeVideoBit();
private:
	bool			misInit;
	std::string		murl;
	std::string		mremoteAddr;
	std::string		modeName;
		//ˮӡ��
	bool				misWaterMark;	//�Ƿ���ˮӡ��
	uint32				mwaterMarkOriHashIdx;
	HASH				mwaterMarkOriHash;
	uint32				mhashIdx;
	HASH				mhash;
		//�����л�
	int					mautoBitRateMode;					//��֡�������ʡ���֡�����ʱ�־
	bool				misChangeVideoBit;					//�Ƿ��л�������
	uint32				mbeginVideoChangeTime;				//�ײ���֡
	uint32				mendVideoChangeTime;	            //�ײ���֡
	uint32				mfirstLowBitRateTimeStamp;			//�ײ������������ʱ����������л���������ʱ��ƥ����ѷ���λ��
	std::vector<HASH>	mhashLowBitRate;					//������hash
	std::vector<uint32>	mhashIdxLowBitRate;					//������hashIdx
	int32				mhashLowBitRateIdx;                 //��ǰ����������
	int32				mmaxLowBitRateIdx;                  //�����������
	int32				mchangeLowBitRateTimes;             //�����л�����
	int64				mchangeBitRateTT;
	std::string			mlowBitRateUrl;
	HASH				mlowBitRateHash;
	int64				mcreateBitRateTaskTT;
	HASH				mcreateBitRateHash;
	int					mautoBitRateFactor;					//��̬������ϵ��
	int					mautoFrameFactor;					//��̬��֡ϵ��
		//�����л���֤ʱ���ƽ��
	uint32				mlastVideoChangeBitRateTimestamp;   //�л�����ǰ�����Ƶʱ���
	int32				mlastVideoChangeBitRateRelativeTimestamp;   //�л����ʺ�lastVideoChangeBitRateTimestamp �� �л����ʺ��һ֡��Ƶ֡ʱ�������ֵ
	bool				mneedChangeVideoTimestamp;                  //�л����ʺ���Ҫ����ʱ�����־
	uint32				mlastAudioChangeBitRateTimestamp;	        //�л�����ǰ�����Ƶʱ���
	int32				mlastAudioChangeBitRateRelativeTimestamp;	//�л����ʺ�lastVideoChangeBitRateTimestamp �� �л����ʺ��һ֡��Ƶ֡ʱ�������ֵ
	bool				mneedChangeAudioTimestamp;					//�л����ʺ���Ҫ����ʱ�����־
		//��֡����
	uint32				mno1AudioTimestamp;					//��֡��Ƶʱ���
	uint32				mno1VideoTimestamp;					//��֡��Ƶʱ���
	int64				mconnectTimestamp;					//����ʱ���
	bool				mtransCodeNeedDropVideo;			//�Ƿ���Ҫ��֡
	bool				mtransCodeNoNeedDropVideo;			//�Ƿ���Ҫ��֡
	int64				mlastDiffer;						//�ϴ�ʱ�����
	int32				mloseBufferTimes;					//Ԥ�⿨�ٴ���
	int64				mloseBufferInterval;				//��¼ͳ�Ƽ��ʱ��
	uint32				mlastVideoTimestamp;
	uint32				mbeginDropVideoTimestamp;
		//��֡�����л�
	std::queue<int>		mhistoryDropList;
	int					mhistoryDropTotal;
	int					mhistoryDropNum;
	int					mhistoryDropTime;
	int64				mhistoryDropTT;
};

#endif