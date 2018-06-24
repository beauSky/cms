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
#ifndef __CMS_STATIC_COMMON_H__
#define __CMS_STATIC_COMMON_H__
#include <common/cms_type.h>
#include <string>

#define PACKET_ONE_TASK_DOWNLOAD	0x00
#define PACKET_ONE_TASK_UPLOAD		0x01
#define PACKET_ONE_TASK_MEDIA		0x02
#define PACKET_ONE_TASK_MEM			0x03

#define PACKET_CONN_ADD				0x01
#define PACKET_CONN_DEL				0x02
#define PACKET_CONN_DATA			0x03

struct OneTaskPacket
{
	int	packetID;
};

struct OneTaskDownload 
{
	int		packetID;
	HASH	hash;
	int32	downloadBytes;					//�����ֽ���
	bool	isRemove;
};

struct OneTaskUpload 
{
	int		packetID;
	HASH	hash;
	int32	uploadBytes;					//�ϴ��ֽ���
	int		connAct;
};

struct OneTaskMeida 
{
	int				packetID;
	HASH			hash;
	int32			videoFramerate;			//��Ƶ֡��
	int32			audioFramerate;			//��Ƶ֡��
	int32			audioSamplerate;		//��Ƶ������
	int32			mediaRate;				//ֱ��������
	int32			width;					//��Ƶ��
	int32			height;					//��Ƶ��
	std::string		videoType;				//��Ƶ����
	std::string		audioType;				//��Ƶ����
	std::string		remoteAddr;				//�Զ�ip:port
	std::string		url;					//url��ַ
	bool			isUdp;				    //�Ƿ���udp����
};

struct OneTaskMem 
{
	int		packetID;
	HASH	hash;
	int64	totalMem;
};

struct OneTask
{
	std::string		murl;
	int64			mdownloadTotal;		//����ͳ�������ٶ�
	int64			mdownloadTick;
	int64			mdownloadSpeed;
	uint64			mdownloadTT;

	int64			muploadTotal;		//����ͳ���ϴ��ٶ�
	int64			muploadTick;
	int64			muploadSpeed;
	uint64			muploadTT;

	int32			mmediaRate;			//ֱ��������
	int32			mvideoFramerate;	//��Ƶ֡��
	int32			maudioFramerate;	//��Ƶ֡��
	int32			maudioSamplerate;	//��Ƶ������
	int32			miWidth;			//��Ƶ��
	int32			miHeight;			//��Ƶ��
	std::string		mvideoType;			//��Ƶ����
	std::string		maudioType;			//��Ƶ����

	int32			mtotalConn;			//������ǰ������
	std::string		mreferer;			//refer

	int64			mtotalMem;			//��ǰ��������ռ���ڴ�

	time_t			mttCreate;
	std::string		mremoteAddr;

	bool			misUDP;
};

struct CpuInfo 
{
	long long user;
	long long nice;
	long long sys;
	long long idle;
};

typedef struct
{
	/** 01 */ char interface_name[128]; /** ����������eth0 */

	/** �������� */
	/** 02 */ unsigned long receive_bytes;             /** ���������յ����ֽ��� */
	/** 03 */ unsigned long receive_packets;
	/** 04 */ unsigned long receive_errors;
	/** 05 */ unsigned long receive_dropped;
	/** 06 */ unsigned long receive_fifo_errors;
	/** 07 */ unsigned long receive_frame;
	/** 08 */ unsigned long receive_compressed;
	/** 09 */ unsigned long receive_multicast;

	/** �������� */
	/** 10 */ unsigned long transmit_bytes;             /** �������ѷ��͵��ֽ��� */
	/** 11 */ unsigned long transmit_packets;
	/** 12 */ unsigned long transmit_errors;
	/** 13 */ unsigned long transmit_dropped;
	/** 14 */ unsigned long transmit_fifo_errors;
	/** 15 */ unsigned long transmit_collisions;
	/** 16 */ unsigned long transmit_carrier;
	/** 17 */ unsigned long transmit_compressed;        
}net_info_t;


OneTask *newOneTask();
//static ����
void makeOneTaskDownload(HASH &hash,int32 downloadBytes,bool isRemove);										//ͳ����������
void makeOneTaskupload(HASH	&hash,int32 uploadBytes,int connAct);											//ͳ���ϴ�����
void makeOneTaskMedia(HASH	&hash,int32 videoFramerate,int32 audioFramerate,int32 iWidth, int32 iHeight,	//ͳ������ý����Ϣ
	int32 audioSamplerate,int32 mediaRate,std::string videoType,std::string audioType, std::string url, 
	std::string remoteAddr,	bool isUdp);
void makeOneTaskMem(HASH &hash,int64 totalMem);																//ͳ���ڴ�ռ��
#endif
