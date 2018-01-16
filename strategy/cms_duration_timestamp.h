/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: ���û������/kisslovecsh@foxmail.com

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
#ifndef __CMS_DURATION_TIMESTAMP_H__
#define __CMS_DURATION_TIMESTAMP_H__
#include <common/cms_type.h>
#include <string>

class CDurationTimestamp
{
public:
	CDurationTimestamp();
	~CDurationTimestamp();

	bool	isInit();
	void	init(std::string remoteAddr,std::string modeName,std::string url);
	void	setResetTimestamp(bool is);
	uint32  resetTimestamp(uint32 timestamp,bool isVideo);
	void	resetDeltaTimestamp(uint32 timestamp);
	uint32	keepTimestampIncrease(bool isVideo,uint32 timestamp);
private:
	bool			misInit;
	std::string		murl;
	std::string		mremoteAddr;
	std::string		modeName;

	//����ʱ���
	bool	misStreamResetTimestamp;  //��������ʱ�����־
	uint32	mvideoResetTimestamp;     //��¼��Ƶʱ���
	uint32	maudioResetTimestamp;     //��¼��Ƶʱ���
		//�������ӱ���ʱ��������� ֻ���û����ţ�����ת�롢��Ƭ����ͼ��¼�ơ�ת�Ʋ�ʹ�ã�������
	uint32	mdeltaVideoTimestamp;
	uint32	mdeltaLastVideoTimestamp;
	uint32	mdeltaAudioTimestamp;
	uint32	mdeltaLastAudioTimestamp;
};
#endif
