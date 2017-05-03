#include <conn/cms_conn_rtmp.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <ev/cms_ev.h>
#include <protocol/cms_flv.h>
#include <common/cms_char_int.h>
#include <taskmgr/cms_task_mgr.h>
#include <libev/ev.h>
#include <enc/cms_sha1.h>
#include <assert.h>
#include <stdlib.h>
using namespace std;

CConnRtmp::CConnRtmp(RtmpType rtmpType,CReaderWriter *rw,std::string pullUrl,std::string pushUrl)
{
	char remote[23] = {0};
	rw->remoteAddr(remote,sizeof(remote));
	mremoteAddr = remote;
	size_t pos = mremoteAddr.find(":");
	if (pos == string::npos)
	{
		mremoteIP = mremoteAddr;
	}
	else
	{
		mremoteIP = mremoteAddr.substr(0,pos);
	}	
	mrdBuff = new CBufferReader(rw,128*1024);
	assert(mrdBuff);
	mwrBuff = new CBufferWriter(rw,128*1024);
	assert(mwrBuff);
	mrw = rw;
	mrtmp = new CRtmpProtocol(this,rtmpType,mrdBuff,mwrBuff,rw,mremoteAddr);
	murl = pullUrl;
	mloop = NULL;
	mwatcherReadIO = NULL;
	mwatcherWriteIO = NULL;
	mvideoType = 0xFF;
	maudioType = 0xFF;
	misChangeMediaInfo = false;
	miFirstPlaySkipMilSecond = 0;
	misResetStreamTimestamp = false;	
	misNoTimeout = false;
	miLiveStreamTimeout = 1000*60*10;
	miNoHashTimeout = 1000*3;
	misRealTimeStream = false;
	mllCacheTT = 1000*15;
	misPublish = false;
	misPlay = false;
	mllIdx = 0;
	misPushFlv = false;
	mflvTrans = new CFlvTransmission(mrtmp);
	misStop = false;
	mjustTickOld = 0;
	mjustTick = 0;
	mrtmpType = rtmpType;
	misPush = false;

	if (!pullUrl.empty())
	{
		makeHash();
		LinkUrl linkUrl;
		if (parseUrl(pullUrl,linkUrl))
		{
			mHost = linkUrl.host;
		}		
	}
	if (!pushUrl.empty())
	{
		setPushUrl(pushUrl);
	}
}

CConnRtmp::~CConnRtmp()
{	
	logs->debug("######### %s [CConnRtmp::~CConnRtmp] %s rtmp %s enter ",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
	if (mloop)
	{
		if (mwatcherReadIO)
		{
			ev_io_stop(mloop,mwatcherReadIO);
			delete mwatcherReadIO;
			logs->debug("######### %s [CConnRtmp::~CConnRtmp] %s rtmp %s stop read io ",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		if (mwatcherWriteIO)
		{
			ev_io_stop(mloop,mwatcherWriteIO);
			delete mwatcherWriteIO;

			logs->debug("######### %s [CConnRtmp::~CConnRtmp] %s rtmp %s stop write io ",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
	}	
	delete mflvTrans;
	delete mrtmp;
	delete mrdBuff;
	delete mwrBuff;
	mrw->close();
	delete mrw;
}

int CConnRtmp::doit()
{
	if (mrtmpType == RtmpClient2Publish)
	{
		if (!CTaskMgr::instance()->pushTaskAdd(mpushHash,this))
		{
			logs->warn("######### %s [CConnRtmp::doit] %s rtmp %s push task is exist %s ",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),mstrPushUrl.c_str());
			return CMS_ERROR;
		}
		else
		{
			misPush = true;
		}
	}
	return CMS_OK;
}

int CConnRtmp::stop(std::string reason)
{
	logs->debug("%s [CConnRtmp::stop] %s rtmp %s enter stop ",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
	//该接口可能被被调用两次,当reason为空是表示正常结束
	if (reason.empty() && misPushFlv)
	{
		Slice *s = newSlice();
		copy2Slice(s);
		s->mhHash = mHash;
		s->misPushTask = misPublish;
		s->misRemove = true;
		CFlvPool::instance()->push(mHashIdx,s);		
	}
	if (misPlay || misPublish)
	{
		CTaskMgr::instance()->pullTaskDel(mHash);
	}
	if (misPush)
	{
		CTaskMgr::instance()->pushTaskDel(mpushHash);
	}
	if (!reason.empty())
	{
		logs->error("%s [CConnRtmp::stop] %s rtmp %s stop with reason: %s ***",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),reason.c_str());
	}
	misStop = true;
	return CMS_OK;
}

int CConnRtmp::handleEv(FdEvents *fe)
{
	if (misStop)
	{
		return CMS_ERROR;
	}
	
	if (fe->events & EventWrite || fe->events & EventWait2Write)
	{
		return doWrite(fe->events & EventWait2Write);
	}
	if (fe->events & EventRead || fe->events & EventWait2Read)
	{
		return doRead();
	}
	if (fe->events & EventJustTick)
	{
		justTick();
	}
	if (fe->events & EventErrot)
	{
		logs->error("%s [CConnRtmp::handleEv] %s rtmp %s handlEv recv event error ***",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int CConnRtmp::doRead()
{
	//logs->debug("%s [CConnRtmp::doRead] rtmp %s doRead",
	//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
	return mrtmp->want2Read();
}

int CConnRtmp::doWrite(bool isTimeout)
{
	//logs->debug("%s [CConnRtmp::doWrite] rtmp %s doWrite",
	//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
	mjustTick++;
	int ret = mrtmp->want2Write(isTimeout);
	mjustTick--;
	return ret;
}

void CConnRtmp::justTick()
{
	if (mjustTick == 0)
	{
		mjustTickOld = mjustTick;
	}
	else
	{
		logs->debug("%s [CConnRtmp::justTick] rtmp %s o no,mjustTick=%llu,mjustTickOld=%llu",
				mremoteAddr.c_str(),mrtmp->getRtmpType().c_str(),mjustTick,mjustTickOld);
	}
}

void CConnRtmp::setEVLoop(struct ev_loop *loop)
{
	mloop = loop;
}

struct ev_loop *CConnRtmp::evLoop()
{
	return mloop;
}

struct ev_io *CConnRtmp::evReadIO()
{
	if (mwatcherReadIO == NULL)
	{
		mwatcherReadIO = new (ev_io);
		ev_io_init(mwatcherReadIO, readEV, mrw->fd(), EV_READ);
		ev_io_start(mloop, mwatcherReadIO);
		//测试
		/*mwatcherTimer = new(ev_timer);
		mwatcherTimer->data = (void *)mrw->fd();
		ev_init(mwatcherTimer,justTickEV);  
		ev_timer_set(mwatcherTimer,0,10);  
		ev_timer_start(mloop,mwatcherTimer); */
	}
	return mwatcherReadIO;
}

struct ev_io *CConnRtmp::evWriteIO()
{
	if (mwatcherWriteIO == NULL)
	{
		mwatcherWriteIO = new (ev_io);
		ev_io_init(mwatcherWriteIO, writeEV, mrw->fd(), EV_WRITE);
		ev_io_start(mloop, mwatcherWriteIO);
		//测试
		/*mwatcherTimer = new(ev_timer);
		mwatcherTimer->data = (void *)mrw->fd();
		ev_init(mwatcherTimer,justTickEV);  
		ev_timer_set(mwatcherTimer,0,10);  
		ev_timer_start(mloop,mwatcherTimer); */
	}
	return mwatcherWriteIO;
}

int CConnRtmp::decodeMessage(RtmpMessage *msg)
{
	bool isSave = false;
	int ret = CMS_OK;
	assert(msg);
	if (msg->dataLen == 0 || msg->buffer == NULL)
	{
		return CMS_OK;
	}
	switch (msg->msgType)
	{
	case MESSAGE_TYPE_CHUNK_SIZE:
		{
			ret = mrtmp->decodeChunkSize(msg);
		}
		break;
	case MESSAGE_TYPE_ABORT:
		{
			logs->debug("%s [CConnRtmp::decodeMessage] %s rtmp %s received abort message,discarding.",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;
	case MESSAGE_TYPE_ACK:
		{
			//logs->debug("%s [CConnRtmp::decodeMessage] rtmp %s received ack message,discarding.",
			//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;
	case MESSAGE_TYPE_USER_CONTROL:
		{
			ret = mrtmp->handleUserControlMsg(msg);
		}
		break;
	case MESSAGE_TYPE_WINDOW_SIZE:
		{
			ret = mrtmp->decodeWindowSize(msg);
		}
		break;
	case MESSAGE_TYPE_BANDWIDTH:
		{
			ret = mrtmp->decodeBandWidth(msg);
		}
		break;
	case MESSAGE_TYPE_DEBUG:
		{
			logs->debug("%s [CConnRtmp::decodeMessage] %s rtmp %s received debug message,discarding.",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;	
	case MESSAGE_TYPE_AMF3_SHARED_OBJECT:
		{
			logs->debug("%s [CConnRtmp::decodeMessage] %s rtmp %s received amf3 share object message,discarding.",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;	
	case MESSAGE_TYPE_INVOKE:
		{
			ret = mrtmp->decodeAmf03(msg,false);
		}
		break;
	case MESSAGE_TYPE_AMF0_SHARED_OBJECT:
		{
			logs->debug("%s [CConnRtmp::decodeMessage] %s rtmp %s received amf0 share object message,discarding.",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;
	case MESSAGE_TYPE_AMF0:
		{
			ret = mrtmp->decodeAmf03(msg,false);
		}
		break;
	case MESSAGE_TYPE_AMF3:
		{
			ret = mrtmp->decodeAmf03(msg,true);
		}
	case MESSAGE_TYPE_FLEX:
		{
			logs->debug("%s [CConnRtmp::decodeMessage] %s rtmp %s received type flex message,discarding.",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		}
		break;
	case MESSAGE_TYPE_AUDIO:
		{
			//logs->debug("%s [CConnRtmp::decodeMessage] rtmp %s received audio message,discarding.",
			//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
			ret = decodeAudio(msg,isSave);
		}
		break;
	case MESSAGE_TYPE_VIDEO:
		{
			//logs->debug("%s [CConnRtmp::decodeMessage] rtmp %s received video message,discarding.",
			//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
			ret = decodeVideo(msg,isSave);
		}
		break;	
	case MESSAGE_TYPE_STREAM_VIDEO_AUDIO:
		{
			//logs->debug("%s [CConnRtmp::decodeMessage] rtmp %s received video audio message,discarding.",
			//	mremoteAddr.c_str(),mrtmp->getRtmpType().c_str());
			ret = decodeVideoAudio(msg);
		}
		break;
	default:
		logs->error("*** %s [CConnRtmp::decodeMessage] %s rtmp %s received unkown message type %d ***",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),msg->msgType);
	}
	if (!isSave)
	{
		delete []msg->buffer;
		msg->buffer = NULL;
		msg->bufLen = 0;
	}
	return ret;
}

int  CConnRtmp::decodeVideo(RtmpMessage *msg,bool &isSave)
{
	if (msg->dataLen <= 0)
	{
		return CMS_OK;
	}
	mrtmp->shouldCloseNodelay();
	byte vType = byte(msg->buffer[0] & 0x0F);
	if (mvideoType == 0xFF || mvideoType != vType)
	{
		if (mvideoType == 0xFF)
		{
			logs->info("%s [CConnRtmp::decodeVideo] %s rtmp %s first video type %s",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),getVideoType(vType).c_str());
		}
		else
		{
			logs->info("%s [CConnRtmp::decodeVideo] %s rtmp %s first video type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),getVideoType(mvideoType).c_str(),getVideoType(vType).c_str());
		}
		mvideoType = vType;
		misChangeMediaInfo = true;
	}
	bool isKeyFrame = false;
	FlvPoolDataType dataType = DATA_TYPE_VIDEO;
	if (vType == 0x02)
	{
		if (msg->dataLen <= 1)
		{
			return CMS_OK; //空帧
		}
	}
	else if (vType == 0x07)
	{
		if (msg->dataLen <= 5)
		{
			return CMS_OK; //空帧
		}
		if (msg->buffer[0] == 0x17)
		{
			isKeyFrame = true;
			if (msg->buffer[1] == 0x00)
			{
				dataType = DATA_TYPE_FIRST_VIDEO;
				miVideoFrameRate = 30;
				misChangeMediaInfo = true;
			}
			else if (msg->buffer[1] == 0x01)
			{

			}
		}
	}
	Slice *s = newSlice();
	copy2Slice(s);	
	s->mData = msg->buffer;
	s->miDataLen = msg->dataLen;
	s->mhHash = mHash;
	s->miDataType = dataType;
	s->misKeyFrame = isKeyFrame;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	if (dataType != DATA_TYPE_FIRST_VIDEO)
	{		
		mllIdx++;
		s->muiTimestamp = msg->absoluteTimestamp;
	}
	CFlvPool::instance()->push(mHashIdx,s);
	isSave = true;
	misPushFlv = true;
	return CMS_OK;
}

int  CConnRtmp::decodeAudio(RtmpMessage *msg,bool &isSave)
{
	if (msg->dataLen <= 0)
	{
		return CMS_OK;//空帧
	}
	mrtmp->shouldCloseNodelay();
	byte aType = byte(msg->buffer[0]>>4 & 0x0F);
	if (maudioType == 0xFF || maudioType != aType)
	{
		if (maudioType == 0xFF)
		{
			logs->info("%s [CConnRtmp::decodeAudio] %s rtmp %s first audio type %s",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),getAudioType(aType).c_str());
		}
		else
		{
			logs->info("%s [CConnRtmp::decodeAudio] %s rtmp %s first audio type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),getAudioType(maudioType).c_str(),getAudioType(aType).c_str());
		}
		maudioType = aType;
		misChangeMediaInfo = true;
	}
	FlvPoolDataType dataType = DATA_TYPE_AUDIO;
	if (aType == 0x02 ||
		aType == 0x04 ||
		aType == 0x05 ||
		aType == 0x06 ){
		if (msg->dataLen <= 1)
		{
			return CMS_OK;//空帧
		}
	} 
	else if (aType >= 0x0A )
	{
		if (msg->dataLen >= 2 &&
			msg->buffer[1] == 0x00)
		{
			if (msg->dataLen <= 2)
			{
				return CMS_OK;//空帧
			}
			dataType = DATA_TYPE_FIRST_AUDIO;
			if (msg->dataLen >= 4)
			{
				miAudioSamplerate = getAudioSampleRates(msg->buffer);
				miAudioFrameRate = getAudioFrameRate(miAudioSamplerate);
				logs->info("%s [CConnRtmp::decodeAudio] %s rtmp %s audio sampleRate=%d,audio frame rate=%d",
					mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),miAudioSamplerate,miAudioFrameRate);
			}
		}
	}
	Slice *s = newSlice();
	copy2Slice(s);	
	s->mData = msg->buffer;
	s->miDataLen = msg->dataLen;
	s->mhHash = mHash;
	s->miDataType = dataType;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	if (dataType != DATA_TYPE_FIRST_AUDIO)
	{
		mllIdx++;
		s->muiTimestamp = msg->absoluteTimestamp;
	}
	CFlvPool::instance()->push(mHashIdx,s);
	isSave = true;
	misPushFlv = true;
	return CMS_OK;
}

int  CConnRtmp::decodeVideoAudio(RtmpMessage *msg)
{
	uint32 uiHandleLen = 0;
	uint32 uiOffset;
	uint32 tagLen;
	uint32 frameLen;
	char pp[4];
	char *p;
	bool isSave;
	while (uiHandleLen < msg->dataLen)
	{
		uint32 dataType = (uint32)msg->buffer[0];
		tagLen = bigUInt24(msg->buffer+uiHandleLen+1);
		if (msg->dataLen < uiHandleLen+11+tagLen+4)
		{
			logs->error("%s [CConnRtmp::decodeVideoAudio] %s rtmp %s video audio check fail ***",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
			return CMS_ERROR;
		}
		//时间戳
		p = msg->buffer+uiHandleLen+1+3;
		pp[2] = p[0];
		pp[1] = p[1];
		pp[0] = p[2];
		pp[3] = p[3];
		
		uiOffset = uiHandleLen+11+tagLen;
		frameLen = bigUInt32(msg->buffer+uiOffset);
		if (frameLen != tagLen+11)
		{
			logs->error("%s [CConnRtmp::decodeVideoAudio] %s rtmp %s video audio tagLen=%u,frameLen=%u ***",
				mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),tagLen,frameLen);
			return CMS_ERROR;
		}
		if (tagLen == 0)
		{
			uiHandleLen += (11+tagLen+1);
			continue;
		}
		RtmpMessage *rm = new RtmpMessage;
		rm->buffer = new char[tagLen];
		memcpy(rm->buffer,msg->buffer+uiHandleLen+11,tagLen);
		rm->dataLen = tagLen;
		rm->msgType = dataType;
		rm->streamId = msg->streamId;
		rm->absoluteTimestamp = littleInt32(pp);
		isSave = false;
		if (dataType == MESSAGE_TYPE_VIDEO)
		{
			if (decodeVideo(rm,isSave) == CMS_ERROR)
			{
				delete[] rm->buffer;
				delete rm;
				return CMS_ERROR;
			}
		}
		else if (dataType == MESSAGE_TYPE_AUDIO)
		{
			if (decodeAudio(rm,isSave) == CMS_ERROR)
			{
				delete[] rm->buffer;
				delete rm;
				return CMS_ERROR;
			}
		}
		if (!isSave)
		{
			delete[] rm->buffer;
		}
		delete rm;
		uiHandleLen += (11+tagLen+1);
	}
	return CMS_OK;
}


int CConnRtmp::decodeMetaData(amf0::Amf0Block *block)
{
	std::string strRtmpContent = amf0::amf0BlockDump(block);
	logs->info("%s [CConnRtmp::decodeMetaData] %s rtmp %s received metaData: %s",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),strRtmpContent.c_str());
	string strRtmpData = amf0::amf0Block2String(block);

	string rate;
	if (amf0::amf0Block5Value(block,"videodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate = atoi(rate.c_str()); //视频码率
	}
	if (amf0::amf0Block5Value(block,"audiodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate += atoi(rate.c_str()); //音频码率
	}
	string value;
	if (amf0::amf0Block5Value(block,"width",value) != amf0::AMF0_TYPE_NONE)
	{
		miWidth = atoi(value.c_str()); //视频高度
	}
	if (amf0::amf0Block5Value(block,"height",value) != amf0::AMF0_TYPE_NONE)
	{
		miHeight += atoi(value.c_str()); //视频宽度
	}
	//采样率
	if (amf0::amf0Block5Value(block,"framerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miVideoFrameRate = atoi(rate.c_str()); 
	}
	if (amf0::amf0Block5Value(block,"audiosamplerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miAudioSamplerate = atoi(rate.c_str()); 
		miAudioFrameRate = getAudioFrameRate(miAudioSamplerate);
	}
	misChangeMediaInfo = true;
	logs->info("%s [CConnRtmp::decodeMetaData] %s rtmp %s stream media rate=%d,width=%d,height=%d,video framerate=%d,audio samplerate=%d,audio framerate=%d",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),strRtmpContent.c_str(),miMediaRate,miWidth,miHeight,miVideoFrameRate,miAudioSamplerate,miAudioFrameRate);

	string strMetaData = amf0::amf0Block2String(block);
	Slice *s = newSlice();
	copy2Slice(s);
	s->mData = new char[strMetaData.length()];
	memcpy(s->mData,strMetaData.c_str(),strMetaData.length());
	s->miDataLen = strMetaData.length();
	s->mhHash = mHash;
	s->misMetaData = true;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	CFlvPool::instance()->push(mHashIdx,s);
	misPushFlv = true;
	return CMS_OK;
}

int CConnRtmp::decodeSetDataFrame(amf0::Amf0Block *block)
{
	amf0::amf0BlockRemoveNode(block,0);	
	amf0::Amf0Data *data = amf0::amf0BlockGetAmf0Data(block,1);
	amf0::Amf0Data *objectEcma = amf0::amf0EcmaArrayNew();
	amf0::Amf0Node *node = amf0::amf0ObjectFirst(data);
	while (node)
	{
		amf0::Amf0Data *nodeName = amf0ObjectGetName(node);
		amf0::Amf0Data *nodeData = amf0ObjectGetData(node);
		amf0::Amf0Data *cloneData = amf0::amf0DataClone(nodeData);
		amf0::amf0ObjectAdd(objectEcma,(const char *)nodeName->string_data.mbstr,cloneData);
		node = amf0::amf0ObjectNext(node);
	}

	amf0::Amf0Block *blockMetaData = amf0::amf0BlockNew();
	amf0::amf0BlockPush(blockMetaData,amf0::amf0StringNew((amf0::uint8 *)"onMetaData",10));
	amf0::amf0BlockPush(blockMetaData,objectEcma);

	std::string strMetaData = amf0::amf0Block2String(blockMetaData);
	std::string strRtmpContent = amf0::amf0BlockDump(blockMetaData);

	logs->info("%s [CConnRtmp::decodeSetDataFrame] %s rtmp %s received metaData: %s",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),strRtmpContent.c_str());
	string rate;
	if (amf0::amf0Block5Value(block,"videodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate = atoi(rate.c_str()); //视频码率
	}
	if (amf0::amf0Block5Value(block,"audiodatarate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miMediaRate += atoi(rate.c_str()); //音频码率
	}
	string value;
	if (amf0::amf0Block5Value(block,"width",value) != amf0::AMF0_TYPE_NONE)
	{
		miWidth = atoi(value.c_str()); //视频高度
	}
	if (amf0::amf0Block5Value(block,"height",value) != amf0::AMF0_TYPE_NONE)
	{
		miHeight = atoi(value.c_str()); //视频宽度
	}
	//采样率
	if (amf0::amf0Block5Value(block,"framerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miVideoFrameRate = atoi(rate.c_str()); 
	}
	if (amf0::amf0Block5Value(block,"audiosamplerate",rate) != amf0::AMF0_TYPE_NONE)
	{
		miAudioSamplerate = atoi(rate.c_str());
		miAudioFrameRate = getAudioFrameRate(miAudioSamplerate);
	}
	amf0::amf0BlockRelease(blockMetaData);
	misChangeMediaInfo = true;	
	logs->info("%s [CConnRtmp::decodeSetDataFrame] %s rtmp %s stream media rate=%d,width=%d,height=%d,video framerate=%d,audio samplerate=%d,audio framerate=%d",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),strRtmpContent.c_str(),miMediaRate,miWidth,miHeight,miVideoFrameRate,miAudioSamplerate,miAudioFrameRate);

	Slice *s = newSlice();
	copy2Slice(s);
	s->mData = new char[strMetaData.length()];
	memcpy(s->mData,strMetaData.c_str(),strMetaData.length());
	s->miDataLen = strMetaData.length();
	s->mhHash = mHash;
	s->misMetaData = true;
	s->mllIndex = mllIdx;
	s->misPushTask = misPublish;
	CFlvPool::instance()->push(mHashIdx,s);
	misPushFlv = true;
	return CMS_OK;
}

void CConnRtmp::copy2Slice(Slice *s)
{
	if (misChangeMediaInfo)
	{
		s->misHaveMediaInfo = true;
		s->miVideoFrameRate = miVideoFrameRate;
		s->miVideoRate = miVideoRate;
		s->miAudioFrameRate = miAudioFrameRate;
		s->miAudioRate = miAudioRate;		
		s->miFirstPlaySkipMilSecond = miFirstPlaySkipMilSecond;
		s->misResetStreamTimestamp = misResetStreamTimestamp;
		s->mstrUrl = murl;
		s->miMediaRate = miMediaRate;
		s->miVideoRate = miVideoRate;
		s->miAudioRate = miAudioRate;
		s->miVideoFrameRate = miVideoFrameRate;
		s->miAudioFrameRate = miAudioFrameRate;
		s->misNoTimeout = misNoTimeout;
		s->mstrVideoType = getVideoType(mvideoType);
		s->mstrAudioType = getAudioType(maudioType);
		s->miLiveStreamTimeout = miLiveStreamTimeout;
		s->miNoHashTimeout = miNoHashTimeout;
		s->mstrRemoteIP = mremoteIP;
		s->mstrHost = mHost;
		s->misRealTimeStream = misRealTimeStream;
		s->mllCacheTT = mllCacheTT;

		misChangeMediaInfo = false;
	}
}

int CConnRtmp::doTransmission()
{
	return mflvTrans->doTransmission();
}

std::string CConnRtmp::getUrl()
{
	return murl;
}

std::string CConnRtmp::getPushUrl()
{
	return mstrPushUrl;
}

std::string CConnRtmp::getRemoteIP()
{
	return mremoteIP;
}

void CConnRtmp::setUrl(std::string url)
{
	if (!url.empty())
	{
		LinkUrl linkUrl;
		if (parseUrl(url,linkUrl))
		{
			mHost = linkUrl.host;
		}
		murl = url;
		makeHash();
	}
	else
	{
		logs->error("***** %s [CConnRtmp::setUrl] %s rtmp %s url is empty *****",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
	}
}

void CConnRtmp::setPushUrl(std::string url)
{
	if (!url.empty())
	{
		mstrPushUrl = url;
		makePushHash();
	}
	else
	{
		logs->error("***** %s [CConnRtmp::setPushUrl] %s rtmp %s url is empty *****",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
	}
}

int CConnRtmp::setPublishTask()
{	
	if (!CTaskMgr::instance()->pullTaskAdd(mHash,this))
	{
		logs->error("***** %s [CConnRtmp::setPublishTask] %s rtmp %s publish task is exist *****",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		return CMS_ERROR;
	}
	misPublish = true;
	mrw->setReadBuffer(1024*32);
	return CMS_OK;
}

int CConnRtmp::setPlayTask()
{	
	if (!CTaskMgr::instance()->pullTaskAdd(mHash,this))
	{
		logs->error("***** %s [CConnRtmp::setPlayTask] %s rtmp %s task is exist *****",
			mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str());
		return CMS_ERROR;
	}
	misPlay = true;
	mrw->setReadBuffer(1024*32);
	return CMS_OK;
}

void CConnRtmp::tryCreateTask()
{
	if (!CTaskMgr::instance()->pullTaskIsExist(mHash))
	{
		CTaskMgr::instance()->createTask(murl,"","",CREATE_ACT_PULL,false,false);
	}
}

void CConnRtmp::makeHash()
{
	string hashUrl = readHashUrl(murl);
	CSHA1 sha;
	sha.write(hashUrl.c_str(), hashUrl.length());
	string strHash = sha.read();
	mHash = HASH((char *)strHash.c_str());
	mstrHash = hash2Char(mHash.data);
	mHashIdx = CFlvPool::instance()->hashIdx(mHash);
	logs->debug("%s [CConnRtmp::makeHash] %s rtmp %s hash url %s,hash=%s",
		mremoteAddr.c_str(),murl.c_str(),mrtmp->getRtmpType().c_str(),hashUrl.c_str(),mstrHash.c_str());
	mflvTrans->setHash(mHashIdx,mHash);
}

void CConnRtmp::makePushHash()
{
	string hashUrl = readMajorUrl(mstrPushUrl);
	CSHA1 sha;
	sha.write(hashUrl.c_str(), hashUrl.length());
	string strHash = sha.read();
	mpushHash = HASH((char *)strHash.c_str());
	mstrHash = hash2Char(mpushHash.data);
	logs->debug("%s [CConnRtmp::makePushHash] %s rtmp %s push hash url %s,hash=%s",
		mremoteAddr.c_str(),mstrPushUrl.c_str(),mrtmp->getRtmpType().c_str(),hashUrl.c_str(),mstrHash.c_str());
}
