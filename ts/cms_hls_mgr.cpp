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
#include <ts/cms_hls_mgr.h>
#include <ts/cms_ts_chunk.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <flvPool/cms_flv_var.h>
#include <flvPool/cms_flv_pool.h>
#include <time.h>
#include <assert.h>

//TEST
std::map<unsigned long,unsigned long> gmapSendTakeTime;
int64 gSendTakeTimeTT = getTimeUnix();
CLock gSendTakeTime;
//TEST end

int64 guid[NUM_OF_THE_HLS_MGR];
CLock guidLock[NUM_OF_THE_HLS_MGR];

int64 getUid(uint32 i)
{
	int64 uid;
	guidLock[i].Lock();
	uid = guid[i];
	guid[i] += NUM_OF_THE_HLS_MGR;
	guidLock[i].Unlock();
	return uid;
}

SSlice *newSSlice()
{
	SSlice *ss = new SSlice();
	ss->mionly = 0;
	ss->msliceRange = 0;  //切片时长
	ss->msliceLen = 0;    //切片大小
	ss->msliceIndex = 0;  //切片序号
	ss->msliceStart = 0;  //切片开始时间戳
	atomicInc(ss);
	return ss;
}

void atomicInc(SSlice *s)
{
	__sync_add_and_fetch(&s->mionly,1);//当数据超时，且没人使用时，删除
}

void atomicDec(SSlice *s)
{
	if (__sync_sub_and_fetch(&s->mionly,1) == 0)//当数据超时，且没人使用时，删除
	{
		TsChunkArray *tca = NULL;
		TsChunk *tc = NULL;
		std::vector<TsChunkArray *>::iterator itTca = s->marray.begin();
		for (; itTca != s->marray.end();)
		{
			tca = *itTca;
			freeTsChunkArray(tca);
			delete tca;
			itTca = s->marray.erase(itTca);
		}
	}
}

void cmsTagReadTimer(void *t)
{
	cms_timer *ct = (cms_timer *)t;
	CMissionMgr::instance()->push(ct->uid%NUM_OF_THE_HLS_MGR,ct);
}

CMission::CMission(HASH &hash,uint32 hashIdx,std::string url,
				   int tsDuration,int tsNum,int tsSaveNum)
{
	mhash = hash;
	mhashIdx = hashIdx;
	murl = url;

	muid = ::getUid(hashIdx%NUM_OF_THE_HLS_MGR);
	mcmsReadTimeOutDo = 0;
	mreadTimer = mallcoCmsTimer();
	cms_timer_init(mreadTimer,0,cmsTagReadTimer,false,muid);

	mtsDuration = tsDuration;
	mtsNum = tsNum;
	mtsSaveNum = tsSaveNum;
	msliceCount = 0;    //切片计数
	msliceIndx = 0;     //当前切片的序号

	if (mtsSaveNum < mtsNum)
	{
		logs->warn("[CMission::CMission] %s ts save num %d is less than ts num %d",mtsSaveNum,mtsNum);
		mtsSaveNum = mtsNum+2;
	}

	for (int i =0; i < mtsSaveNum/*+mtsNum*/+1; i++)
	{
		msliceList.push_back(NULL);
	}
	msliceList[0] = newSSlice();

	misStop = false;

	mreadIndex = -1;		//读取的帧的序号

	mreadFAIndex = -1;	//读取的音频首帧的序号
	mreadFVIndex = -1;	//读取的视频首帧的序号

	mFAFlag = false;	//是否读到首帧音频
	mFVFlag = false;	//是否读到首帧视频(SPS/PPS)
	mbTime = 0;			//最后一个切片的生成时间
	mMux = new CSMux(); //转码器

	mdurationtt = new CDurationTimestamp();
}

CMission::~CMission()
{
	if (mMux)
	{
		delete mMux;
	}
	if (mdurationtt)
	{
		delete mdurationtt;
	}
	std::vector<SSlice *>::iterator it = msliceList.begin();
	for (;it != msliceList.end();)
	{
		if (*it != NULL)
		{
			atomicDec(*it);
		}
		it = msliceList.erase(it);
	}
	freeCmsTimer(mreadTimer);
}

int CMission::doFirstVideoAudio(bool isVideo)
{
	int ret = CMS_OK;
	Slice *s = NULL;
	if (CFlvPool::instance()->readRirstVideoAudioSlice(mhashIdx,mhash,&s,isVideo) == FlvPoolCodeError)
	{
		return ret;
	}
	assert(s != NULL);
	byte *outBuf = NULL;
	int  outLen = 0;
	if (isVideo)
	{		
		logs->debug(">>>>>[CMission::doFirstVideoAudio] %s doFirstVideoAudio parse first video",
			murl.c_str());
		mreadFVIndex = s->mllIndex;
		mMux->parseNAL((byte *)s->mData,s->miDataLen,&outBuf,outLen);
		mFVFlag = true;
	}
	else
	{		
		logs->debug(">>>>>[CMission::doFirstVideoAudio] %s doFirstVideoAudio parse first audio",
			murl.c_str());
		mreadFAIndex = s->mllIndex;
		mMux->parseAAC((byte *)s->mData,s->miDataLen,&outBuf,outLen);
		mFAFlag = true;
	}
	if (outBuf)
	{
		delete[] outBuf;
	}
	atomicDec(s);
	return ret;
}

int  CMission::doit(cms_timer *t)
{
	if (t != NULL && t != mreadTimer)
	{
		//xiang tong de uid?
		logs->error("*** [CMission::doit] %s cms timer is not equal ***",
			murl.c_str());
		freeCmsTimer(t);
		return 0;
	}
	if (t)
	{
		assert(mcmsReadTimeOutDo==1);
		mcmsReadTimeOutDo--;
	}
	int ret = 0;
	Slice *s = NULL;
	int flvPoolCode;
	uint32 uiTimestamp = 0;	
	int	sliceNum = 0;
	bool isVideo = false;
	bool isAudio = false;
	do 
	{
		if (CFlvPool::instance()->isFirstVideoChange(mhashIdx,mhash,mreadFVIndex))
		{
			if (doFirstVideoAudio(true) == CMS_ERROR)
			{
				ret = -1;
				break;
			}
		}
		if (CFlvPool::instance()->isFirstAudioChange(mhashIdx,mhash,mreadFAIndex))
		{
			if (doFirstVideoAudio(false) == CMS_ERROR)
			{
				ret = -1;
				break;
			}
		}
		sliceNum = 0;
		flvPoolCode = CFlvPool::instance()->readSlice(mhashIdx,mhash,mreadIndex,&s,sliceNum);
		if (flvPoolCode == FlvPoolCodeError)
		{
			//logs->error("*** [CMission::doit] %s task is missing ***",
			//	murl.c_str());
			//ret = -1;
			break;
		}
		else if (flvPoolCode == FlvPoolCodeOK)
		{		
			if (!mdurationtt->isInit())
			{
				mdurationtt->init("0.0.0.0","flv",murl);
				mdurationtt->setResetTimestamp(false);
			}
			if (s)
			{				
				//首播丢帧 结束
				isVideo = s->miDataType == DATA_TYPE_VIDEO;
				isAudio = s->miDataType == DATA_TYPE_AUDIO;

				uiTimestamp = s->muiTimestamp;				
				//预防时间戳变小的情况
				uiTimestamp = mdurationtt->keepTimestampIncrease(isVideo,uiTimestamp);
				//预防时间戳变小的情况 结束
				if (mreadIndex+1 != s->mllIndex)
				{
					logs->warn("*** [CMission::doit] %s doit drop frame,cur idx %lld,next idx %lld ***",
						murl.c_str(),mreadIndex,s->mllIndex);
				}
				mreadIndex = s->mllIndex;


				byte frameType;
				uint64 pts64;

				TsChunkArray *tca = NULL;

				if (isAudio){
					frameType = 'A';
					mMux->packTS((byte *)s->mData, s->miDataLen,frameType,uiTimestamp,0x01,0x01,&mMux->getAmcc(),Apid,&tca,pts64);
					if (!mFAFlag)
					{
					}
				} 
				else if (isVideo) 
				{
					frameType = 'P';
					if (s->mData[0] == 0x17) 
					{
						frameType = 'I';
					}
					mMux->packTS((byte *)s->mData, s->miDataLen,frameType,uiTimestamp,0x01,0x01,&mMux->getVmcc(),Vpid,&tca,pts64);
					if (!mFVFlag)
					{						
					}
				}
				if (tca) {
					pushData(tca, frameType, pts64);
				}
				atomicDec(s);
				if (ret == CMS_ERROR)
				{
					ret = -1;
					break;
				}
				ret = 1;
			}
			else
			{
				break;
			}
		}
		else if (flvPoolCode == FlvPoolCodeNoData)
		{
			ret = 0;
			break;
		}
		else if (flvPoolCode == FlvPoolCodeRestart)
		{
			mreadIndex = -1;
			mreadFVIndex = -1;
			mreadFAIndex = -1;

			uint32 sendTimestamp = 0;
			if (s != NULL)
			{
				sendTimestamp = s->muiTimestamp;
				atomicDec(s);
			}

			mdurationtt->resetDeltaTimestamp(sendTimestamp);
			logs->debug("[CMission::doit] %s doit task is been restart",
				murl.c_str());
			ret = 0;
			break;
		}
		else
		{
			logs->error("*** [CMission::doit] %s doit unknow FlvPoolCode=%d ***",
				murl.c_str(),flvPoolCode);
			ret = -1;
			break;
		}
	} while (true);
	if (mcmsReadTimeOutDo == 0)
	{
		mcmsReadTimeOutDo++;
		cms_timer_start(mreadTimer);			
	}	
	return ret;
}

void CMission::stop()
{
	
}

/*输入一帧TS数据
-- inbuf 送入的数据缓冲区
-- length 数据大小
-- framtype	帧类型(用于切片)
-- timestamp 时间戳
*/
int  CMission::pushData(TsChunkArray *tca,byte frameType,uint64 timestamp)
{
	assert(tca->msliceSize%TS_CHUNK_SIZE==0);
//  	logs->debug(" [CMission::doit] %s pushData enter,len=%d,frameType=%c,timestamp=%d,msliceList[msliceCount]->msliceStart=%d,mtsDuration*90000=%d",
//  		murl.c_str(),inLen,frameType,timestamp,msliceList[msliceCount]->msliceStart,mtsDuration*90000);
// 	logs->debug(" [CMission::doit] %s pushData (timestamp-msliceList[msliceCount]->msliceStart) > mtsDuration*90000=%s, frameType == 'I'=%s, (mFVFlag == false && frameType == 'A')=%s,frameType=%d,'I'=%d",
// 			murl.c_str(),(timestamp-msliceList[msliceCount]->msliceStart) > mtsDuration*90000?"true":"false",
// 			frameType == 'I'?"true":"false",(mFVFlag == false && frameType == 'A')?"true":"false",frameType,'I');
	SSlice *ss = NULL;
	if ((msliceList[msliceCount]->msliceStart != 0) && timestamp-msliceList[msliceCount]->msliceStart > (uint64)(mtsDuration*90000) && 
		(frameType == 'I' || (mFVFlag == false && frameType == 'A')))
	{
// 		logs->debug(" [CMission::pushData] %s pushData if enter,msliceCount+1=%d,mtsSaveNum=%d,timestamp=%llu,msliceStart=%llu,timestamp-msliceStart=%llu,msliceIndx=%lld,msliceList size=%u",
// 			murl.c_str(),msliceCount+1,mtsSaveNum,timestamp,msliceList[msliceCount]->msliceStart,timestamp-msliceList[msliceCount]->msliceStart,msliceList[msliceCount]->msliceIndex,msliceList.size());
		mbTime = time(NULL);
		if (msliceCount+1 > /*mtsNum+*/mtsSaveNum)
		{			
			msliceIndx++;
			ss = newSSlice();
			ss->msliceIndex = msliceIndx;
			ss->msliceStart = timestamp;
			mMux->packPSI();

			int writeLen = 0;
			TsChunkArray *ttmp = allocTsChunkArray(188*2);
			writeChunk(NULL,0,ttmp,(char *)mMux->getPat(),188,writeLen);
			writeChunk(NULL,0,ttmp,(char *)mMux->getPmt(),188,writeLen);

			ss->msliceLen = 188*2;
			ss->msliceLen += tca->msliceSize;

			ss->marray.push_back(ttmp);
			ss->marray.push_back(tca);

			msliceList.push_back(ss);

			msliceList[msliceCount]->msliceRange = (float)((timestamp-msliceList[msliceCount]->msliceStart)/90000);

			logs->debug("[CMission::pushData] 1 %s pushData one ts succ,timestamp=%d",
				murl.c_str(),timestamp-msliceList[msliceCount-1]->msliceStart);

			std::vector<SSlice *>::iterator it = msliceList.begin();
			ss = *it;
			msliceList.erase(it);
			atomicDec(ss);			
		}
		else
		{
			msliceCount++;
			msliceIndx++;
			ss = newSSlice();
			ss->msliceIndex = msliceIndx;
			ss->msliceStart = timestamp;
			mMux->packPSI();
			
			int writeLen = 0;
			TsChunkArray *ttmp = allocTsChunkArray(188*2);
			writeChunk(NULL,0,ttmp,(char *)mMux->getPat(),188,writeLen);
			writeChunk(NULL,0,ttmp,(char *)mMux->getPmt(),188,writeLen);

			ss->msliceLen = 188*2;
			ss->msliceLen += tca->msliceSize;

			ss->marray.push_back(ttmp);
			ss->marray.push_back(tca);

			msliceList[msliceCount] = ss;

			ss->msliceRange = (float)((timestamp-ss->msliceStart)/90000);

			logs->debug("[CMission::pushData] 2 %s pushData one ts succ,timestamp=%d",
				murl.c_str(),timestamp-msliceList[msliceCount-1]->msliceStart);
		}
	}
	else
	{
		ss = msliceList[msliceCount];
		if (ss->msliceStart == 0)
		{
			ss->msliceStart = timestamp;
		}
		ss->msliceLen += tca->msliceSize;
		ss->marray.push_back(tca);
	}
	return 0;
}

/*获取切片TS
-- Index 切片序号
*/
int  CMission::getTS(int64 idx,SSlice **s)
{
// 	logs->debug("[CMission::getTS] %s getTS idx=%lld,msliceIndx=%lld,msliceList[0]->msliceIndex=%lld",
// 		murl.c_str(),idx,msliceIndx,msliceList[0]->msliceIndex);

	if (idx >= msliceIndx || idx < msliceList[0]->msliceIndex)
	{
		return -1;
	}
	int64 pos = idx - msliceList[0]->msliceIndex;
	*s = msliceList[pos];
	atomicInc(*s);
	return 1;
}
/*获取M3U8列表
-- addr 组装绝对路径的addr
*/
int  CMission::getM3U8(std::string addr,std::string &outData)
{
	size_t end = murl.rfind("/");
	std::string url =  murl.substr(7,end-7);
	outData += "#EXTM3U\n";

	char szData[30] = {0};
	snprintf(szData,sizeof(szData),"%d",mtsDuration+1);
	outData += "#EXT-X-TARGETDURATION:";
	outData +=  szData;
	outData += "\n";

	int pos = 0;
	if (msliceCount <= mtsNum)
	{
		pos = 0;
	} 
	else if (msliceCount >= (mtsSaveNum/* + mtsNum*/))
	{
		pos = mtsSaveNum-mtsNum;
	} 
	else 
	{
		pos = msliceCount - mtsNum;
	}
	snprintf(szData,sizeof(szData),"%lld",msliceList[pos]->msliceIndex);
	outData += "#EXT-X-MEDIA-SEQUENCE:";
	outData += szData;
	outData += "\n";

	for (int64 i = msliceList[pos]->msliceIndex; i < msliceIndx; i++)
	{
 		//logs->debug("\n\nm3u8 url %s ,murl %s,end %d",url.c_str(),murl.c_str(),end);
		snprintf(szData,sizeof(szData),"%0.2f", msliceList[pos]->msliceRange);
		outData += "#EXTINF:";
		outData += szData;
		outData += ",\n";
		outData += "http://";
		outData += addr;
		outData += "/";
		outData += url;
		outData += "/";
		snprintf(szData,sizeof(szData),"%lld",i);
		outData += szData;
		outData += ".ts\n";
		pos++;
	}
	return 1;
}

int64 CMission::getLastTsTime()
{
	return mbTime;
}

int64 CMission::getUid()
{
	return muid;
}

//mgr
CMissionMgr *CMissionMgr::minstance = NULL;
CMissionMgr::CMissionMgr()
{
	for (int i = 0; i < NUM_OF_THE_HLS_MGR; i++)
	{
		guid[i] = i;
		misRunning[i] = false;
	}
}

CMissionMgr::~CMissionMgr()
{

}

void *CMissionMgr::routinue(void *param)
{
	HlsMgrThreadParam *phmtp = (HlsMgrThreadParam *)param;
	CMissionMgr *pInstance = phmtp->pinstance;
	uint32 i = phmtp->i;
	delete phmtp;
	pInstance->thread(i);
	return NULL;
}

void CMissionMgr::thread(uint32 i)
{
	logs->debug("### enter CMissionMgr thread=%d ###", gettid());
	cms_timer *ct;
	bool is;
	while (misRunning[i])
	{
		ct = NULL;
		is = pop(i,&ct);
		if (is)
		{
			tick(i,ct);
		}
		else
		{
			cmsSleep(10);
		}
	}
	logs->debug("### leave CMissionMgr thread=%d ###", gettid());
}

bool CMissionMgr::run()
{
	for (uint32 i = 0; i < NUM_OF_THE_HLS_MGR; i++)
	{
		misRunning[i] = true;
		HlsMgrThreadParam *hmtp = new HlsMgrThreadParam;
		hmtp->pinstance = this;
		hmtp->i = i;
		int res = cmsCreateThread(&mtid[i],routinue,hmtp,false);
		if (res == -1)
		{
			misRunning[i] = false;
			char date[128] = {0};
			getTimeStr(date);
			logs->error("%s ***** file=%s,line=%d cmsCreateThread error *****\n",date,__FILE__,__LINE__);
			return false;
		}		
	}	
	return true;
}

CMissionMgr *CMissionMgr::instance()
{
	if (minstance == NULL)
	{
		minstance = new CMissionMgr();
	}
	return minstance;
}

void CMissionMgr::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}

int	 CMissionMgr::create(uint32 i,HASH &hash,std::string url,int tsDuration,int tsNum,int tsSaveNum)
{
	int ret = CMS_ERROR;
	uint32 ii = i % NUM_OF_THE_HLS_MGR;
	mMissionMapLock[ii].WLock();
	std::map<HASH,CMission *>::iterator it = mMissionMap[ii].find(hash);
	if (it == mMissionMap[ii].end())
	{
		CMission *cm = new CMission(hash,i,url,tsDuration,tsNum,tsSaveNum);
		cm->doit();
		mMissionMap[ii][hash] = cm;
		mMissionUidMap[ii][cm->getUid()] = cm;
		ret = CMS_OK;
		logs->debug("### [CMissionMgr::create] thread idx=%d have hls num %d ###", i,mMissionUidMap[ii].size());
	}
	mMissionMapLock[ii].UnWLock();
	return ret;
}

void CMissionMgr::destroy(uint32 i,HASH &hash)
{
	i = i % NUM_OF_THE_HLS_MGR;
	mMissionMapLock[i].WLock();
	std::map<HASH,CMission *>::iterator it = mMissionMap[i].find(hash);
	if (it != mMissionMap[i].end())
	{
		std::map<int64,CMission *>::iterator it1 = mMissionUidMap[i].find(it->second->getUid());
		if (it1 != mMissionUidMap[i].end())
		{
			mMissionUidMap[i].erase(it1);
		}

		it->second->stop();
		delete it->second;
		mMissionMap[i].erase(it);
	}
	mMissionMapLock[i].UnWLock();
}

int CMissionMgr::readM3U8(uint32 i,HASH &hash,std::string url,std::string addr,std::string &outData,int64 &tt)
{
	i = i % NUM_OF_THE_HLS_MGR;
	int ret = -1;
	mMissionMapLock[i].RLock();
	std::map<HASH,CMission *>::iterator it = mMissionMap[i].find(hash);
	if (it != mMissionMap[i].end())
	{
		it->second->getM3U8(addr,outData);
		tt = it->second->getLastTsTime();
		ret = 1;
	}
	mMissionMapLock[i].UnRLock();
	return ret;
}

int CMissionMgr::readTS(uint32 i,HASH &hash,std::string url,std::string addr,SSlice **ss,int64 &tt)
{
	i = i % NUM_OF_THE_HLS_MGR;
	int ret = -1;
	mMissionMapLock[i].RLock();
	std::map<HASH,CMission *>::iterator it = mMissionMap[i].find(hash);
	if (it != mMissionMap[i].end())
	{
		size_t start = url.rfind("/")+1;
		if (start == std::string::npos)
		{

		}
		size_t end = url.find(".ts");
		if (end == std::string::npos)
		{

		}
		string strIdx = url.substr(start,end);
		int64 llIdx = _atoi64(strIdx.c_str());
		ret = it->second->getTS(llIdx,ss);
	}
	mMissionMapLock[i].UnRLock();
	return ret;
}

void CMissionMgr::release()
{

}

void CMissionMgr::push(uint32 i,cms_timer *t)
{
	i = i % NUM_OF_THE_HLS_MGR;
	mqueueWL[i].Lock();
	mqueueRT[i].push(t);
	mqueueWL[i].Unlock();
}
bool CMissionMgr::pop(uint32 i,cms_timer **t)
{
	i = i % NUM_OF_THE_HLS_MGR;
	bool is = false;
	mqueueWL[i].Lock();
	if (!mqueueRT[i].empty())
	{
		*t = mqueueRT[i].front();
		mqueueRT[i].pop();
		is = true;
	}
	mqueueWL[i].Unlock();
	return is;
}

void CMissionMgr::tick(uint32 i,cms_timer *t)
{
	unsigned long tB = getTickCount();


	i = i % NUM_OF_THE_HLS_MGR;
	assert(t != NULL);
	mMissionMapLock[i].WLock();
	std::map<int64,CMission *>::iterator it = mMissionUidMap[i].find(t->uid);
	if (it != mMissionUidMap[i].end())
	{
		it->second->doit(t);
	}
	mMissionMapLock[i].UnWLock();



	unsigned long tE = getTickCount();
	gSendTakeTime.Lock();
	int64 sendTakeTimeTT = getTimeUnix();
	bool isPrintf = false;
	if (sendTakeTimeTT - gSendTakeTimeTT > 60)
	{
		isPrintf = true;
		gSendTakeTimeTT = sendTakeTimeTT;
	}
	printTakeTime(gmapSendTakeTime,tB,tE,"CMissionMgr",isPrintf);
	gSendTakeTime.Unlock();
}