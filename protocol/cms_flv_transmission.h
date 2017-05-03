#ifndef __CMS_FLV_TRANSMISSION_H__
#define __CMS_FLV_TRANSMISSION_H__
#include <common/cms_type.h>
#include <protocol/cms_rtmp.h>
#include <interface/cms_protocol.h>
#include <strategy/cms_fast_bit_rate.h>
#include <string>

class CRtmpProtocol;
class CFlvTransmission
{
public:
	CFlvTransmission(CProtocol *protocol);
	~CFlvTransmission();
	void setHash(uint32 hashIdx,HASH &hash);
	void setWaterMarkHash(uint32 hashIdx,HASH &hash);
	int  doTransmission();
private:
	int  doMetaData();
	int  doFirstVideoAudio(bool isVideo);
	void getSliceFrameRate();
	CProtocol	*mprotocol;
	HASH		mreadHash;
	uint32      mreadHashIdx;

	bool		misWaterMark;//�Ƿ񲥷�ˮӡԴ��
	HASH		mwaterMarkOriHash;
	uint32		mwaterMarkOriHashIdx;
	std::string	murlWaterMark;

	CFastBitRate *mfastBitRate;
	//����ֱ�������
	int64			mllMetaDataIdx;
	int64			mllFirstVideoIdx;
	int64			mllFirstAudioIdx;
	int64			mllTransIdx;
	bool			misChangeFirstVideo;
	int				mchangeFristVideoTimes;

	int64			mcacheTT; //����ʱ�� ms
	uint32			muiKeyFrameDistance;
	int32			msliceFrameRate;
};
#endif