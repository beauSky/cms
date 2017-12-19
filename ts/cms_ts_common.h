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
#ifndef __CMS_TS_COMMON_H__
#define __CMS_TS_COMMON_H__
#include <common/cms_type.h>

#define PATpid  0
#define PMTpid  0x1000
#define Apid    0x101
#define Vpid    0x100
#define PCRpid  0x100

//FLVͷ������Ϣ
typedef struct _SHead  
{
	byte mversion;
	byte mstreamInfo;   //����Ϣ 4-a 1-v 5-a/v
	int  mlenght;       //ͷ����
}SHead;

//FLV ScriptTag��Ϣ
typedef struct _SDataInfo  
{
	int mduration;         //ʱ��
	int	mwidth;            //��Ƶ���
	int	mheight;           //��Ƶ�߶�
	int	mvideodatarate;    //��Ƶ����
	int	mframerate;        //��Ƶ֡��
	int	mvideocodecid;     //��Ƶ���뷽ʽ
	int	maudiosamplerate;  //��Ƶ������
	int	maudiosamplesize;  //��Ƶ��������
	int	mstereo;           //�Ƿ�Ϊ������
	int	maudiocodecid;     //��Ƶ���뷽ʽ
	int	mfilesize;         //�ļ���С
}SDataInfo;

//FLV Tagͷ������Ϣ
typedef struct _STagHead
{
	byte	mtagType;		//tag����
	int		mdataSize;      //tag����
	uint32	mtimeStamp;		//ʱ���
	int		mstreamId;      //��ID
	int		mdeviation;     //ʱ���ƫ����
}STagHead;

//FLV ��ƵTag��Ϣ
typedef struct _SAudioInfo
{
	byte mcodeType;   //��������
	byte mrate;       //������
	byte mprecision;  //����
	byte maudioType;  //��Ƶ����
}SAudioInfo;

//FLV ��ƵTag��Ϣ
typedef struct _SVideoInfo
{
	byte mframType;  //֡����
	byte mcodeId;    //��������
}SVideoInfo;

//����Tag��Ϣ
typedef struct _STagInfo
{
	STagHead	mhead;
	byte		mflag;   //v:video a:audio
	SAudioInfo	maudio; 
	SVideoInfo	mvideo;
}STagInfo;

//AAC������Ϣ
typedef struct _SAudioSpecificConfig
{
	byte	mObjectType;       //5
	byte	mSamplerateIndex;  //4
	byte	mChannels;         //4
	byte	mFramLengthFlag;   //1
	byte	mDependOnCCoder;   //1
	byte	mExtensionFlag;    //1
}SAudioSpecificConfig;
#endif
