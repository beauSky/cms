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
#ifndef __CMS_JITTER_H__
#define __CMS_JITTER_H__
#include <common/cms_type.h>
#include <string>

class CJitter
{
public:
	CJitter();
	~CJitter();

	bool	isInit();
	void	init(std::string remoteAddr,std::string modeName,std::string url);
	void	reset();
	void	jitterTimestamp(float tba,float tbv,float lastTimestamp,
		uint32 curTimestamp,bool isVideo,float &rlastTimestamp,uint32 &rcurTimestamp);
	uint32  judgeNeedJitter(bool isVideo,uint32 absoluteTimestamp);
	bool	countVideoAudioFrame(bool isVideo,int64 tk,uint32 absoluteTimestamp);
	float	getAudioJitterFrameRate();
	float	getVideoJitterFrameRate();
	void	setAudioJitterFrameRate(float audioJitterFrameRate);
	void	setVideoJitterFrameRate(float videoJitterFrameRate);
	float	getAudioJitterCountFrameRate();
	void	setAudioFrameRate(int audioFrameRate);
	void	setVideoFrameRate(int videoFrameRate);
	void	setOpenJitter(bool is);
	void	setOpenForceJitter(bool is);
	bool	isOpenJitter();
	bool	isForceJitter();
private:
	bool			misInit;
	std::string		murl;
	std::string		mremoteAddr;
	std::string		modeName;
	int				mvideoFrameRate;
	int				maudioFrameRate;
	bool			misOpenJitter;
	bool			misForceJitter;
	int				mvideoaudioTimestampIllegal;		//����Ƶʱ�������
	int				mvideoJitterTimestampIllegal;		//��Ƶʱ����Ƿ���������
	float			mabsLastVideoTimestamp;				//���һ�θ��µ���Ƶ����ʱ���
	float			mvideoJitterFrameRate;				//��Ƶ֡��
	int				maudioJitterTimestampIllegal;		//��Ƶʱ����Ƿ���������
	float			mabsLastAudioTimestamp;				//���һ�θ��µ���Ƶ����ʱ���
	float			maudioJitterFrameRate;				//��Ƶ֡��
	int				mvideoJitterNum;					//�ۼƾ�������֡
	float			mvideoJitterDetalTotal;				//�ۼƾ���ʱ��
	int				maudioJitterNum;					//�ۼƾ�������֡
	float			maudioJitterDetalTotal;				//�ۼƾ���ʱ��
	bool			misJitterAudio;						//�Ƿ���Ҫ������Ƶ֡
	bool			misJitterVideo;						//�Ƿ���Ҫ������Ƶ֡
		//��Ϊͳ����Ƶ֡��,ֻ��ͳ�Ƶ�֡�ʺͻ�ȡ����֡����ӽ�ʱ���Ż�����ʱ�����
	float			mvideoJitterCountFrameRate;			//ͳ�Ƶ���Ƶ֡��
	uint32			mvideoBeginJitterTimestamp;			//
	float			mvideoJitterGrandTimestmap;			//��Ƶ��������
	bool			misCountVideoFrameRate;     
	bool			misLegallVideoFrameRate;    
		//��Ϊͳ����Ƶ֡��,ֻ��ͳ�Ƶ�֡�ʺͻ�ȡ����֡����ӽ�ʱ���Ż�����ʱ�����
	float			maudioJitterCountFrameRate;
	uint32			maudioBeginJitterTimestamp; 
	float			maudioJitterGrandTimestmap;			//��Ƶ��������
	bool			misCountAudioFrameRate;     
	bool			misLegallAudioFrameRate;    
	int				mvideoAudioJitterBalance;			//����Ƶ��ƽ��������ʾ��Ƶ��������ʾ��Ƶ
};
#endif
