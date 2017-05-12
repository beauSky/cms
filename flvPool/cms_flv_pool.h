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
		int				mionly;				//0 ��ʾû��ʹ�ã�����0��ʾ���ڱ�ʹ�ô���
		FlvPoolDataType miDataType;			//��������
		bool            misHaveMediaInfo;   //�Ƿ����޸Ĺ�����Ϣ
		bool			misPushTask;
		bool			misNoTimeout;
		bool			misMetaData;		//�������Ƿ���metaData
		bool			misRemove;			//ɾ�������־
		int				miNotPlayTimeout;	//��ʱʱ�䣬����
		uint32			muiTimestamp;	    //��slice���ݶ�Ӧrtmp��ʱ���
		
		int64			mllP2PIndex;		//p2p������
		int64			mllIndex;           //��slice��Ӧ�����к�
		int64			mllOffset;			//ƫ��λ�ã�������
		int64			mllStartTime;		//����ʼʱ��
		char			*mData;				//����
		int             miDataLen;
		HASH			mhMajorHash;		//����ת�����񣬸�hash��ʾԴ��hash
		HASH			mhHash;				//��ǰ����hash
		std::string     mstrUrl;
		bool			misKeyFrame;
		int				miMediaRate;
		int				miVideoRate;		//��Ƶ����
		int				miAudioRate;		//��Ƶ����
		int				miVideoFrameRate;	//��Ƶ֡��
		int				miAudioFrameRate;	//��Ƶ֡��
		std::string     mstrVideoType;		//��Ƶ����
		std::string     mstrAudioType;		//��Ƶ����
		int				miAudioChannelID;	//��������ƵID

		std::string     mstrReferUrl;
		int64			mllCacheTT;					//����ʱ�� ����
		int				miPlayStreamTimeout;		//���û���ų�ʱʱ��	
		bool			misRealTimeStream ;			//�Ƿ�����µ����ݷ���
		int				miFirstPlaySkipMilSecond;	//�ײ���֡ʱ��
		int				miAutoBitRateMode;			//��̬��֡ģʽ(0/1/2)
		int				miAutoBitRateFactor;		//��̬������ϵ��
		int				miAutoFrameFactor;			//��̬��֡ϵ��
		int				miBufferAbsolutely;			//buffer�ٷֱ�
		bool			misResetStreamTimestamp;

		int				miLiveStreamTimeout;
		int				miNoHashTimeout;
		std::string     mstrRemoteIP;
		std::string     mstrHost;
};

struct TTandKK
{
	int64			mllIndex;		//��ͨ��Ƶ����
	int64			mllKeyIndex;
	uint32			muiTimestamp;	//ʱ���
};

struct StreamSlice 
{
		//������ʱ����
		int64						maxRelativeDuration;
		int64						minRelativeDuration;
		CRWlock						mLock;
		//��ʱ�������
		std::vector<TTandKK *>		msliceTTKK;
		int64						mllNearKeyFrameIdx;
		uint32						muiTheLastVideoTimestamp;
		
		bool						misPushTask;
		bool						mnoTimeout;			//�����Ƿ񲻳�ʱ
		std::string					mstrUrl;
		std::string					mstrReferUrl;
		HASH						mhMajorHash;
		int							miNotPlayTimeout;	//��ʱʱ�䣬����
		int64						mllAccessTime;		//��¼ʱ�������һ��ʱ��û���û����ʣ�ɾ��
		int64						mllCreateTime;		//���񴴽�ʱ��

		std::vector<Slice *>		mavSlice;
		std::vector<int64>			mavSliceIdx;
		std::vector<int64>			mvKeyFrameIdx;		//�ؼ�֡λ��
		std::vector<int64>			mvP2PKeyFrameIdx;	//�ؼ�֡λ��
		std::map<int64,int64>		mp2pIdx2PacketIdx;
		int							miVideoFrameCount;
		int							miAudioFrameCount;
		int							miMediaRate;
		int							miVideoRate;		//��Ƶ����
		int							miAudioRate;		//��Ƶ����
		int							miVideoFrameRate;	//��Ƶ֡��
		int							miAudioFrameRate;	//��Ƶ֡��
		std::string					mstrVideoType;		//��Ƶ����
		std::string					mstrAudioType;		//��Ƶ����
		uint32						muiKeyFrameDistance;
		uint32						muiLastKeyFrameDistance;

		int64						mllLastSliceIdx;		
	
		Slice						*mfirstVideoSlice;
		int64						mllFirstVideoIdx          ;
		Slice						*mfirstAudioSlice;
		int64						mllFirstAudioIdx;
		bool						misH264;		
		int64						mllVideoAbsoluteTimestamp;	//���ڼ��㻺������
		int64						mllAudioAbsoluteTimestamp;	//���ڼ��㻺������		
		bool						misHaveMetaData;
		int64						mllMetaDataIdx;
		int64						mllMetaDataP2PIdx;

		bool						misNoTimeout;		

		int64						mllLastMemSize;
		int64						mllMemSize;
		int64						mllMemSizeTick;
		Slice						*mmetaDataSlice;
		int64						mllCacheTT;					//����ʱ�� ����
		int							miPlayStreamTimeout;		//���û���ų�ʱʱ��
		bool						misRealTimeStream ;			//�Ƿ�����µ����ݷ���
		int							miFirstPlaySkipMilSecond;	//�ײ���֡ʱ��
		int							miAutoBitRateMode;			//��̬��֡ģʽ(0/1/2)
		int							miAutoBitRateFactor;		//��̬������ϵ��
		int							miAutoFrameFactor;			//��̬��֡ϵ��
		int							miBufferAbsolutely;			//buffer�ٷֱ�
		bool						misResetStreamTimestamp;

		//��Ե�Ż��õ��ı������Ͽ�ʱ��״̬
		bool						misNeedJustTimestamp;
		bool						misRemove;
		bool						misHasBeenRemove;
		int64						mllRemoveTimestamp;
		uint32						muiLastVideoTimestamp;
		uint32						muiLastAudioTimestamp;
		uint32						muiLast2VideoTimestamp;
		uint32						muiLast2AudioTimestamp;

		int64						mllUniqueID;

		//���Ʋ���Ч
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

	std::set<HASH>			msetHash[FLV_POOL_COUNT]; //��ʱ��¼����¼������û���ʾ�ɾ��
	CLock					msetHashLock[FLV_POOL_COUNT];

	cms_thread_t			mtid[FLV_POOL_COUNT];
};
#endif
