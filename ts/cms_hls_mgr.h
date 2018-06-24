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
#ifndef __CMS_HLS_MGR_H__
#define __CMS_HLS_MGR_H__
#include <common/cms_type.h>
#include <common/cms_var.h>
#include <core/cms_thread.h>
#include <ts/cms_ts.h>
#include <ev/cms_ev.h>
#include <core/cms_lock.h>
#include <strategy/cms_duration_timestamp.h>
#include <app/cms_app_info.h>
#include <string>
#include <vector>
#include <map>
#include <queue>


typedef struct _SSlice {
	int		mionly;		  //0 ��ʾû��ʹ�ã�����0��ʾ���ڱ�ʹ�ô���
	float	msliceRange;  //��Ƭʱ��
	int		msliceLen;    //��Ƭ��С
	int64	msliceIndex;  //��Ƭ���
	uint64	msliceStart;  //��Ƭ��ʼʱ���
	std::vector<TsChunkArray *> marray;	  //��Ƭ����
}SSlice;

SSlice *newSSlice();
void atomicInc(SSlice *s);
void atomicDec(SSlice *s);

void cmsTagReadTimer(void *t);

class CMission 
{
public:
	CMission(HASH &hash,uint32 hashIdx,std::string url,
		int tsDuration,int tsNum,int tsSaveNum);
	~CMission();

	int  doFirstVideoAudio(bool isVideo);
	int  doit(cms_timer *t = NULL);
	void stop();
	int  pushData(TsChunkArray *tca,byte frameType,uint64 timestamp);
	int  getTS(int64 idx,SSlice **s);
	int  getM3U8(std::string addr,std::string &outData);
	int64 getLastTsTime();
	int64 getUid();
private:
	int     mcmsReadTimeOutDo;
	cms_timer *mreadTimer;
	int64      muid;

	HASH	mhash;			//����ʶ�������hashֵ
	uint32  mhashIdx;		//
	std::string murl;		//ƴ���õ�URL

	int		mtsDuration;    //������Ƭʱ��
	int		mtsNum;         //��Ƭ���޸���
	int		mtsSaveNum;     //���汣������Ƭ����
	std::vector<SSlice *> msliceList; //��Ƭ�б�
	int		msliceCount;    //��Ƭ����
	int64	msliceIndx;     //��ǰ��Ƭ�����

	bool	misStop;		//������������Э��

	int64	mreadIndex;		//��ȡ��֡�����

	int64	mreadFAIndex;	//��ȡ����Ƶ��֡�����
	int64	mreadFVIndex;	//��ȡ����Ƶ��֡�����

	bool	mFAFlag;	//�Ƿ������֡��Ƶ
	bool	mFVFlag;	//�Ƿ������֡��Ƶ(SPS/PPS)
	int64	mbTime;		//���һ����Ƭ������ʱ��
	CSMux	*mMux;      //ת����
	TsChunkArray *mlastTca;//��ʡ�ռ�

	uint64  mullTransUid;

	CDurationTimestamp *mdurationtt;
};

class CMissionMgr
{
public:
	CMissionMgr();
	~CMissionMgr();

	static void *routinue(void *param);
	void thread(uint32 i);
	bool run();
	void stop();

	static CMissionMgr *instance();
	static void freeInstance();
	/*����һ������
	-- idx hash��Ӧ��������,��Ƭ�ڲ���Ҫ����
	-- hash �����ϣ
	-- url����url
	-- tsDuration һ����Ƭts��ʱ��
	-- tsNum ��m3u8��tsƬ��
	-- tsSaveNum ��Ƭģ�黺��tsƬ��
	*/
	int	 create(uint32 i,HASH &hash,std::string url,int tsDuration,int tsNum,int tsSaveNum);
	/*����һ������
	-- hash �����ϣ
	*/
	void destroy(uint32 i,HASH &hash);
	/*���������ȡm3u8��ts
	-- hash �����ϣ
	-- url m3u8����ts�ĵ�ַ
	*/
	int  readM3U8(uint32 i,HASH &hash,std::string url,std::string addr,std::string &outData,int64 &tt);
	int  readTS(uint32 i,HASH &hash,std::string url,std::string addr,SSlice **ss,int64 &tt);
	/*���������ͷţ�����������һЩ��ʱ���ݵĻ���*/
	void release();
	void tick(uint32 i,cms_timer *t);
	void push(uint32 i,cms_timer *t);
	bool pop(uint32 i,cms_timer **t);
private:
	static CMissionMgr *minstance;
	cms_thread_t mtid[APP_ALL_MODULE_THREAD_NUM];	
	bool misRunning[APP_ALL_MODULE_THREAD_NUM];
	std::map<HASH,CMission *> mMissionMap[APP_ALL_MODULE_THREAD_NUM];			//�����б�
	std::map<int64,CMission *> mMissionUidMap[APP_ALL_MODULE_THREAD_NUM];			//�����б�
	CRWlock					  mMissionMapLock[APP_ALL_MODULE_THREAD_NUM];

	std::map<HASH,int64>	  mMissionSliceCount[APP_ALL_MODULE_THREAD_NUM];	//�������Ƭ��¼
	CRWlock					  mMissionSliceCountLock[APP_ALL_MODULE_THREAD_NUM];

	//��ʱ
	std::queue<cms_timer *> mqueueRT[APP_ALL_MODULE_THREAD_NUM];
	CLock mqueueWL[APP_ALL_MODULE_THREAD_NUM];
};

struct HlsMgrThreadParam 
{
	CMissionMgr *pinstance;
	uint32 i;
};

#endif
