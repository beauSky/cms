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
#ifndef __CMS_FLV_TRANSMISSION_H__
#define __CMS_FLV_TRANSMISSION_H__
#include <common/cms_type.h>
#include <protocol/cms_rtmp.h>
#include <interface/cms_protocol.h>
#include <strategy/cms_fast_bit_rate.h>
#include <strategy/cms_duration_timestamp.h>
#include <strategy/cms_first_play.h>
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

	bool		misWaterMark;//是否播放水印源流
	HASH		mwaterMarkOriHash;
	uint32		mwaterMarkOriHashIdx;
	std::string	murlWaterMark;

	CFastBitRate		*mfastBitRate;
	CDurationTimestamp	*mdurationtt;
	CFirstPlay			*mfirstPlay;
	//发送直播流相关
	int64			mllMetaDataIdx;
	int64			mllFirstVideoIdx;
	int64			mllFirstAudioIdx;
	int64			mllTransIdx;
	bool			misChangeFirstVideo;
	int				mchangeFristVideoTimes;

	int64			mcacheTT; //缓存时间 ms
	uint32			muiKeyFrameDistance;
	int32			msliceFrameRate;
};
#endif