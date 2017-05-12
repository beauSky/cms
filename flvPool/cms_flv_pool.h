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
#ifndef __CMS_FLV_POOL_H__
#define __CMS_FLV_POOL_H__
#include <protocol/cms_flv.h>
#include <common/cms_type.h>
#include <core/cms_lock.h>
#include <core/cms_thread.h>
#include <strategy/cms_fast_bit_rate.h>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <set>
#include <assert.h>

#ifndef FLV_POOL_COUNT
#define FLV_POOL_COUNT 8
#endif

enum FlvPoolCode
{
	FlvPoolCodeError = -1,
	FlvPoolCodeOK,
	FlvPoolCodeNoData,
	FlvPoolCodeRestart,
	FlvPoolCodeTaskNotExist
};

enum FlvPoolDataType
{
	DATA_TYPE_NONE              = -0x01,
	DATA_TYPE_AUDIO             = 0x00,
	DATA_TYPE_VIDEO             = 0x01,
	DATA_TYPE_VIDEO_AUDIO       = 0x02,
	DATA_TYPE_FIRST_AUDIO       = 0x03,
	DATA_TYPE_FIRST_VIDEO       = 0x04,
	DATA_TYPE_FIRST_VIDEO_AUDIO = 0x05,
	DATA_TYPE_DATA_SLICE        = 0X06
};

#define FLV_TAG_AUDIO		0x08
#define FLV_TAG_VIDEO		0x09
#define FLV_TAG_SCRIPT		0x12

struct Slice 
{
		int				mionly;				//0 表示没被使用，大于0表示正在被使用次数
		FlvPoolDataType miDataType;			//数据类型
		bool            misHaveMediaInfo;   //是否有修改过流信息
		bool			misPushTask;
		bool			misNoTimeout;
		bool			misMetaData;		//该数据是否是metaData
		bool			misRemove;			//删除任务标志
		int				miNotPlayTimeout;	//超时时间，毫秒
		uint32			muiTimestamp;	    //该slice数据对应rtmp的时间戳
		
		int64			mllP2PIndex;		//p2p索引号
		int64			mllIndex;           //该slice对应的序列号
		int64			mllOffset;			//偏移位置（保留）
		int64			mllStartTime;		//任务开始时间
		char			*mData;				//数据
		int             miDataLen;
		HASH			mhMajorHash;		//对于转码任务，该hash表示源流hash
		HASH			mhHash;				//当前任务hash
		std::string     mstrUrl;
		bool			misKeyFrame;
		int				miMediaRate;
		int				miVideoRate;		//视频码率
		int				miAudioRate;		//音频码率
		int				miVideoFrameRate;	//视频帧率
		int				miAudioFrameRate;	//音频帧率
		std::string     mstrVideoType;		//视频类型
		std::string     mstrAudioType;		//音频类型
		int				miAudioChannelID;	//连麦流音频ID

		std::string     mstrReferUrl;
		int64			mllCacheTT;					//缓存时间 毫秒
		int				miPlayStreamTimeout;		//多久没播放超时时间	
		bool			misRealTimeStream ;			//是否从最新的数据发送
		int				miFirstPlaySkipMilSecond;	//首播丢帧时长
		int				miAutoBitRateMode;			//动态丢帧模式(0/1/2)
		int				miAutoBitRateFactor;		//动态变码率系数
		int				miAutoFrameFactor;			//动态丢帧系数
		int				miBufferAbsolutely;			//buffer百分比
		bool			misResetStreamTimestamp;

		int				miLiveStreamTimeout;
		int				miNoHashTimeout;
		std::string     mstrRemoteIP;
		std::string     mstrHost;
};

struct TTandKK
{
	int64			mllIndex;		//普通视频数据
	int64			mllKeyIndex;
	uint32			muiTimestamp;	//时间戳
};

struct StreamSlice 
{
		//两个临时变量
		int64						maxRelativeDuration;
		int64						minRelativeDuration;
		CRWlock						mLock;
		//按时间戳查找
		std::vector<TTandKK *>		msliceTTKK;
		int64						mllNearKeyFrameIdx;
		uint32						muiTheLastVideoTimestamp;
		
		bool						misPushTask;
		bool						mnoTimeout;			//任务是否不超时
		std::string					mstrUrl;
		std::string					mstrReferUrl;
		HASH						mhMajorHash;
		int							miNotPlayTimeout;	//超时时间，毫秒
		int64						mllAccessTime;		//记录时间戳，若一段时间没有用户访问，删除
		int64						mllCreateTime;		//任务创建时间

		std::vector<Slice *>		mavSlice;
		std::vector<int64>			mavSliceIdx;
		std::vector<int64>			mvKeyFrameIdx;		//关键帧位置
		std::vector<int64>			mvP2PKeyFrameIdx;	//关键帧位置
		std::map<int64,int64>		mp2pIdx2PacketIdx;
		int							miVideoFrameCount;
		int							miAudioFrameCount;
		int							miMediaRate;
		int							miVideoRate;		//视频码率
		int							miAudioRate;		//音频码率
		int							miVideoFrameRate;	//视频帧率
		int							miAudioFrameRate;	//音频帧率
		std::string					mstrVideoType;		//视频类型
		std::string					mstrAudioType;		//音频类型
		uint32						muiKeyFrameDistance;
		uint32						muiLastKeyFrameDistance;

		int64						mllLastSliceIdx;		
	
		Slice						*mfirstVideoSlice;
		int64						mllFirstVideoIdx          ;
		Slice						*mfirstAudioSlice;
		int64						mllFirstAudioIdx;
		bool						misH264;		
		int64						mllVideoAbsoluteTimestamp;	//用于计算缓存数据
		int64						mllAudioAbsoluteTimestamp;	//用于计算缓存数据		
		bool						misHaveMetaData;
		int64						mllMetaDataIdx;
		int64						mllMetaDataP2PIdx;

		bool						misNoTimeout;		

		int64						mllLastMemSize;
		int64						mllMemSize;
		int64						mllMemSizeTick;
		Slice						*mmetaDataSlice;
		int64						mllCacheTT;					//缓存时间 毫秒
		int							miPlayStreamTimeout;		//多久没播放超时时间
		bool						misRealTimeStream ;			//是否从最新的数据发送
		int							miFirstPlaySkipMilSecond;	//首播丢帧时长
		int							miAutoBitRateMode;			//动态丢帧模式(0/1/2)
		int							miAutoBitRateFactor;		//动态变码率系数
		int							miAutoFrameFactor;			//动态丢帧系数
		int							miBufferAbsolutely;			//buffer百分比
		bool						misResetStreamTimestamp;

		//边缘才会用到的保存流断开时的状态
		bool						misNeedJustTimestamp;
		bool						misRemove;
		bool						misHasBeenRemove;
		int64						mllRemoveTimestamp;
		uint32						muiLastVideoTimestamp;
		uint32						muiLastAudioTimestamp;
		uint32						muiLast2VideoTimestamp;
		uint32						muiLast2AudioTimestamp;

		int64						mllUniqueID;

		//边推才有效
		int							miLiveStreamTimeout;
		int							miNoHashTimeout;

		std::string					mstrRemoteIP;
		std::string					mstrHost;
};


Slice *newSlice();
StreamSlice *newStreamSlice();

void atomicInc(Slice *s);
void atomicDec(Slice *s);

class CFlvPool
{
public:
	CFlvPool();
	~CFlvPool();

	static void *routinue(void *param);
	void thread(uint32 i);
	bool run();

	static CFlvPool *instance();
	static void freeInstance();

	void stop();
	uint32 hashIdx(HASH &hash);
	void push(uint32 i,Slice *s);
	bool pop(uint32 i,Slice **s);
	int  readRirstVideoAudioSlice(uint32 i,HASH &hash,Slice **s,bool isVideo);	
	int  readSlice(uint32 i,HASH &hash,int64 &llIdx,Slice **s,int &sliceNum,bool isTrans = false );
	bool isHaveMetaData(uint32 i,HASH &hash);
	bool isExist(uint32 i,HASH &hash);
	bool isFirstVideoChange(uint32 i,HASH &hash,int64 &videoIdx);
	bool isFirstAudioChange(uint32 i,HASH &hash,int64 &audioIdx);
	bool isMetaDataChange(uint32 i,HASH &hash,int64 &metaIdx);
	int  readMetaData(uint32 i,HASH &hash,Slice **s);
	int  getFirstVideo(uint32 i,HASH &hash,Slice **s);
	int  getFirstAudio(uint32 i,HASH &hash,Slice **s);
	bool isH264(uint32 i,HASH &hash);
	int64 getMinIdx(uint32 i,HASH &hash);
	int64 getMaxIdx(uint32 i,HASH &hash);
	int	  getMediaRate(uint32 i,HASH &hash);
	void  updateAccessTime(uint32 i,HASH &hash);
	int   getFirstPlaySkipMilSecond(uint32 i,HASH &hash);
	uint32 getDistanceKeyFrame(uint32 i,HASH &hash);
	int   getVideoFrameRate(uint32 i,HASH &hash);
	int   getAudioFrameRate(uint32 i,HASH &hash);

	int			readBitRateMode(uint32 i,HASH &hash);
	std::string readChangeBitRateSuffix(uint32 i,HASH &hash);
	std::string readCodeSuffix(uint32 i,HASH &hash);

	bool seekKeyFrame(uint32 i,HASH &hash,uint32 &tt,int64 &transIdx);
	bool mergeKeyFrame(char *desc,int descLen,char *key,int keyLen,char **src,int32 &srcLen,std::string url);
	int64  getCacheTT(uint32 i,HASH &hash);
	uint32 getKeyFrameDistance(uint32 i,HASH &hash);
	int	   getAutoBitRateFactor(uint32 i,HASH &hash);
	int	   getAutoFrameFactor(uint32 i,HASH &hash);
private:
	void handleSlice(uint32 i,Slice *s); 
	void clear();
	void delHash(uint32 i,HASH &hash);
	void addHash(uint32 i,HASH &hash);
	void checkTimeout();
	bool isTimeout(uint32 i,HASH &hash);
	void getRelativeDuration(StreamSlice *ss,Slice *s,bool isNewSlice,
		int64 &maxRelativeDuration,int64 &minRelativeDuration);


	bool					misRun;
	static CFlvPool			*minstance;

	CLock					mqueueLock[FLV_POOL_COUNT];	
	std::queue<Slice *>		mqueueSlice[FLV_POOL_COUNT];	

	CRWlock					mhashSliceLock[FLV_POOL_COUNT];
	std::map<HASH,StreamSlice *> mmapHashSlice[FLV_POOL_COUNT];

	std::set<HASH>			msetHash[FLV_POOL_COUNT]; //超时记录，记录任务多久没访问就删除
	CLock					msetHashLock[FLV_POOL_COUNT];

	cms_thread_t			mtid[FLV_POOL_COUNT];
};
#endif
