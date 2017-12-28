#ifndef __CMS_FLV_VAR_H__
#define __CMS_FLV_VAR_H__
#include <common/cms_type.h>
#include <core/cms_lock.h>
#include <string>
#include <vector>
#include <map>

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

#define OFFSET_FIRST_VIDEO_FRAME 0x05

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

	bool			misH264;
	bool			misH265;

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
	//idΨһ�� ���ڷ�������ʱ �ж������Ƿ�����
	uint64						muid;
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
	bool						misH265;
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
#endif
