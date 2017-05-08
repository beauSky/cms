#include <conn/cms_http_c.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <ev/cms_ev.h>
#include <taskmgr/cms_task_mgr.h>
#include <enc/cms_sha1.h>
#include <conn/cms_conn_var.h>
#include <flvPool/cms_flv_pool.h>
#include <protocol/cms_amf0.h>
#include <common/cms_char_int.h>
#include <static/cms_static.h>

ChttpClient::ChttpClient(CReaderWriter *rw,std::string pullUrl,std::string oriUrl,
						 std::string refer,bool isTls)
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
	mhttp = new CHttp(this,mrdBuff,mwrBuff,rw,mremoteAddr,true,isTls);
	mhttp->setUrl(pullUrl);
	mhttp->setOriUrl(oriUrl);
	mhttp->setRefer(refer);
	mhttp->httpRequest()->setRefer(refer);
	mhttp->httpRequest()->setUrl(pullUrl);
	mhttp->httpRequest()->setHeader(HTTP_HEADER_REQ_USER_AGENT,"cms");
	if (!refer.empty())
	{
		mhttp->httpRequest()->setHeader(HTTP_HEADER_REQ_REFERER,refer);
	}
	mhttp->httpRequest()->setHeader(HTTP_HEADER_REQ_ICY_METADATA,"1");

	misRequet = false;
	misDecodeHeader = false;
	
	murl = pullUrl;
	moriUrl = oriUrl;
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
	mllIdx = 0;
	misPushFlv = false;
	misDown8upBytes = false;
	misStop = false;
	misRedirect = false;
	//flv
	misReadTagHeader = false;
	misReadTagBody = false;
	misReadTagFooler = false;
	miReadFlvHeader = 13;
	mtagFlv = NULL;;
	mtagLen = 0;
	mtagReadLen = 0;

	mspeedTick = 0;

	if (!pullUrl.empty())
	{
		makeHash();
		LinkUrl linkUrl;
		if (parseUrl(pullUrl,linkUrl))
		{
			mHost = linkUrl.host;
		}		
	}
}

ChttpClient::~ChttpClient()
{
	logs->debug("######### %s [ChttpClient::~ChttpClient] http enter ",
		mremoteAddr.c_str());
	if (mloop)
	{
		if (mwatcherReadIO)
		{
			ev_io_stop(mloop,mwatcherReadIO);
			delete mwatcherReadIO;
			logs->debug("######### %s [ChttpClient::~ChttpClient] stop read io ",
				mremoteAddr.c_str());
		}
		if (mwatcherWriteIO)
		{
			ev_io_stop(mloop,mwatcherWriteIO);
			delete mwatcherWriteIO;

			logs->debug("######### %s [ChttpClient::~ChttpClient] stop write io ",
				mremoteAddr.c_str());
		}
	}
	if (mtagFlv != NULL)
	{
		delete[] mtagFlv;
	}
	delete mhttp;
	delete mrdBuff;
	delete mwrBuff;	
	mrw->close();
	delete mrw;
}

int ChttpClient::doit()
{
	return CMS_OK;
}

int ChttpClient::handleEv(FdEvents *fe)
{
	if (misStop)
	{
		return CMS_ERROR;
	}

	if (fe->events & EventWrite || fe->events & EventWait2Write)
	{
		if (fe->events & EventWait2Write && fe->watcherCmsTimer !=  mhttp->cmsTimer2Write())
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		else if (fe->events & EventWrite && mwatcherWriteIO != fe->watcherWriteIO)
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		return doWrite(fe->events & EventWait2Write);
	}
	if (fe->events & EventRead || fe->events & EventWait2Read)
	{
		if (fe->events & EventRead && mwatcherReadIO != fe->watcherReadIO)
		{
			//应该是旧的socket号的消息
			return CMS_OK;
		}
		return doRead();
	}
	if (fe->events & EventErrot)
	{
		logs->error("%s [ChttpClient::handleEv] handlEv recv event error ***",
			mremoteAddr.c_str());
		return CMS_ERROR;
	}
	return CMS_OK;
}

int ChttpClient::stop(std::string reason)
{	
	//可能会被调用两次,任务断开时,正常调用一次 reason 为空,
	//主动断开时,会调用,reason 是调用原因
	if (reason.empty())
	{
		logs->debug("%s [ChttpClient::stop] http %s has been stop ",
			mremoteAddr.c_str(),murl.c_str());
		if (misPushFlv)
		{
			Slice *s = newSlice();
			copy2Slice(s);
			s->mhHash = mHash;
			s->misRemove = true;
			CFlvPool::instance()->push(mHashIdx,s);
		}

		if (misDown8upBytes)
		{
			down8upBytes();
			makeOneTaskDownload(mHash,0,true);
		}	

		CTaskMgr::instance()->pullTaskDel(mHash);
		if (misRedirect)
		{
			tryCreateTask();
		}
	}
	
	if (!reason.empty())
	{
		logs->error("%s [ChttpClient::stop] %s stop with reason: %s ***",
			mremoteAddr.c_str(),moriUrl.c_str(),reason.c_str());
	}
	misStop = true;
	return CMS_OK;
}

std::string ChttpClient::getUrl()
{
	return murl;
}

std::string ChttpClient::getPushUrl()
{
	return "";
}

std::string ChttpClient::getRemoteIP()
{
	return mremoteIP;
}

struct ev_loop  *ChttpClient::evLoop()
{
	return mloop;
}

struct ev_io    *ChttpClient::evReadIO()
{
	if (mwatcherReadIO == NULL)
	{
		mwatcherReadIO = new (ev_io);
		ev_io_init(mwatcherReadIO, readEV, mrw->fd(), EV_READ);
		ev_io_start(mloop, mwatcherReadIO);
	}
	return mwatcherReadIO;
}

struct ev_io    *ChttpClient::evWriteIO()
{
	if (mwatcherWriteIO == NULL)
	{
		mwatcherWriteIO = new (ev_io);
		ev_io_init(mwatcherWriteIO, writeEV, mrw->fd(), EV_WRITE);
		ev_io_start(mloop, mwatcherWriteIO);
	}
	return mwatcherWriteIO;
}

void ChttpClient::setEVLoop(struct ev_loop *loop)
{
	mloop = loop;
}

int ChttpClient::doDecode()
{
	int ret = CMS_OK;
	if (!misDecodeHeader)
	{
		misDecodeHeader = true;
		if (mhttp->httpResponse()->getStatusCode() == HTTP_CODE_200 ||
			mhttp->httpResponse()->getStatusCode() == HTTP_CODE_206)
		{
			//succ
			std::string transferEncoding = mhttp->httpResponse()->getHeader(HTTP_HEADER_RSP_TRANSFER_ENCODING);
			if (transferEncoding == HTTP_VALUE_CHUNKED)
			{
				mhttp->setChunked();
			}
		}
		else if (mhttp->httpResponse()->getStatusCode() == HTTP_CODE_301 ||
			mhttp->httpResponse()->getStatusCode() == HTTP_CODE_302 ||
			mhttp->httpResponse()->getStatusCode() == HTTP_CODE_303)
		{
			//redirect
			misRedirect = true;
			mredirectUrl = mhttp->httpResponse()->getHeader(HTTP_HEADER_LOCATION);
			logs->info("%s [ChttpClient::doDecode] http %s redirect %s ",
				mremoteAddr.c_str(),moriUrl.c_str(),mhttp->httpResponse()->getStatusCode(),mredirectUrl.c_str());
			ret = CMS_ERROR;
		}
		else
		{
			//error
			logs->error("%s [ChttpClient::doDecode] http %s code %d rsp %s ***",
				mremoteAddr.c_str(),moriUrl.c_str(),mhttp->httpResponse()->getStatusCode(),
				mhttp->httpResponse()->getResponse().c_str());
			ret = CMS_ERROR;
		}
	}
	return ret;
}

int ChttpClient::doReadData()
{
	char *p;
	int ret = 0;
	int len;
	do 
	{
		//flv header
		while (miReadFlvHeader > 0)
		{
			len = miReadFlvHeader;
			ret = mhttp->read(&p,len);
			if (ret <= 0)
			{
				return ret;
			}
			miReadFlvHeader -= ret;
			//logs->debug("%s [ChttpClient::doReadData] http %s read flv header len %d ",
			//	mremoteAddr.c_str(),moriUrl.c_str(),ret);
		}
		//flv tag
		if (!misReadTagHeader)
		{
			char *tagHeader;
			len = 11;
			ret = mhttp->read(&tagHeader,len); //肯定会读到11字节
			if (ret <= 0)
			{
				return ret;
			}

			//printf("%s [ChttpClient::doReadData] http %s read flv tag header len %d \n",
			//	mremoteAddr.c_str(),moriUrl.c_str(),ret);

			miTagType = (FlvPoolDataType)tagHeader[0];
			mtagLen = (int)bigInt24(tagHeader+1);
			if (mtagLen < 0 || mtagLen > 1024*1024*10)
			{
				logs->error("%s [ChttpClient::doReadData] http %s read tag len %d unexpect,tag type=%d ***",
					mremoteAddr.c_str(),moriUrl.c_str(),mtagLen,miTagType);
				return CMS_ERROR;
			}

			//printf("%s [ChttpClient::doReadData] http %s read flv tag type %d, tag len %d \n",
			//	mremoteAddr.c_str(),moriUrl.c_str(),miTagType,mtagLen);

			p = (char*)&muiTimestamp;
			p[2] = tagHeader[4];
			p[1] = tagHeader[5];
			p[0] = tagHeader[6];
			p[3] = tagHeader[7];
			misReadTagHeader = true;
			mtagFlv = new char[mtagLen];
			mtagReadLen = 0;
		}
		
		if (misReadTagHeader)
		{
			if (!misReadTagBody)
			{
				while (mtagReadLen < mtagLen)
				{
					len = mtagLen-mtagReadLen > 1024*8 ? 1024*8 : mtagLen-mtagReadLen;
					ret = mhttp->read(&p,len);
					if (ret <= 0)
					{
						return ret;
					}
					memcpy(mtagFlv+mtagReadLen,p,len);
					mtagReadLen += len;
				}
				misReadTagBody = true;
				//printf("%s [ChttpClient::doReadData] http %s read flv tag body=%d \n",
				//	mremoteAddr.c_str(),moriUrl.c_str(),mtagReadLen);
			}
			if (!misReadTagFooler)
			{
				char *tagHeaderFooler;
				len = 4;
				ret = mhttp->read(&tagHeaderFooler,len); //肯定会读到4字节
				if (ret <= 0)
				{
					return ret;
				}
				misReadTagFooler = true;
				int tagTotalLen = bigInt32(tagHeaderFooler);
				if (mtagLen+11 != tagTotalLen)
				{
					//警告
					//printf("%s [ChttpClient::doReadData] http %s handle tagTotalLen=%d,mtagLen+11=%d \n",
					//	mremoteAddr.c_str(),moriUrl.c_str(),tagTotalLen,mtagLen+11);
				}
				//printf("%s [ChttpClient::doReadData] http %s handle tagTotalLen=%d,mtagLen=%d \n",
				//	mremoteAddr.c_str(),moriUrl.c_str(),tagTotalLen,mtagLen);
			}

			misReadTagHeader = false;
			misReadTagBody = false;
			misReadTagFooler = false;

			switch (miTagType)
			{
			case 0x08:
				//printf("%s [ChttpClient::doReadData] http %s handle audio tag \n",
				//	mremoteAddr.c_str(),moriUrl.c_str());

				decodeAudio(mtagFlv,mtagLen,muiTimestamp);
				mtagFlv = NULL;
				mtagLen = 0;
				break;
			case 0x09:
				//printf("%s [ChttpClient::doReadData] http %s handle video tag \n",
				//	mremoteAddr.c_str(),moriUrl.c_str());

				decodeVideo(mtagFlv,mtagLen,muiTimestamp);
				mtagFlv = NULL;
				mtagLen = 0;
				break;
			case 0x12:
				//printf("%s [ChttpClient::doReadData] http %s handle metaData tag \n",
				//	mremoteAddr.c_str(),moriUrl.c_str());

				decodeMetaData(mtagFlv,mtagLen);
				mtagFlv = NULL;
				mtagLen = 0;
				break;
			default:
				logs->error("*** %s [ChttpClient::doReadData] http %s read tag type %d unexpect *** \n",
					mremoteAddr.c_str(),moriUrl.c_str(),miTagType);
				return 0;
			}
		}
	} while (1);
	return ret;
}

int ChttpClient::doTransmission()
{
	return CMS_OK;
}

int ChttpClient::sendBefore(const char *data,int len)
{
	return CMS_OK;
}

int ChttpClient::doRead()
{
	return mhttp->want2Read();
}

int ChttpClient::doWrite(bool isTimeout)
{
	return mhttp->want2Write(isTimeout);
}

int  ChttpClient::handle()
{
	return CMS_OK;
}

int	 ChttpClient::handleFlv(int &ret)
{
	return CMS_OK;
}

int ChttpClient::request()
{
	return CMS_OK;
}

void ChttpClient::makeHash()
{
	string hashUrl = readHashUrl(murl);
	CSHA1 sha;
	sha.write(hashUrl.c_str(), hashUrl.length());
	string strHash = sha.read();
	mHash = HASH((char *)strHash.c_str());
	mstrHash = hash2Char(mHash.data);
	mHashIdx = CFlvPool::instance()->hashIdx(mHash);
	logs->debug("%s [ChttpClient::makeHash] %s hash url %s,hash=%s",
		mremoteAddr.c_str(),murl.c_str(),hashUrl.c_str(),mstrHash.c_str());
}

void ChttpClient::copy2Slice(Slice *s)
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

		makeOneTaskMedia(mHash,miVideoFrameRate,miAudioFrameRate,miAudioSamplerate,miMediaRate,s->mstrVideoType,s->mstrAudioType,murl,mremoteAddr);
	}
}

void ChttpClient::tryCreateTask()
{
	if (!CTaskMgr::instance()->pullTaskIsExist(mHash))
	{
		CTaskMgr::instance()->createTask(mredirectUrl,"",moriUrl,mstrRefer,CREATE_ACT_PULL,false,false);
	}
}

int ChttpClient::decodeMetaData(char *data,int len)
{
	amf0::Amf0Block *block = NULL;
	block = amf0::amf0Parse(data,len);	
	std::string strRtmpContent = amf0::amf0BlockDump(block);
	logs->info("%s [ChttpClient::decodeMetaData] http %s received metaData: %s",
		mremoteAddr.c_str(),moriUrl.c_str(),strRtmpContent.c_str());
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
	logs->info("%s [ChttpClient::decodeMetaData] http %s stream media rate=%d,width=%d,height=%d,video framerate=%d,audio samplerate=%d,audio framerate=%d",
		mremoteAddr.c_str(),murl.c_str(),strRtmpContent.c_str(),miMediaRate,miWidth,miHeight,miVideoFrameRate,miAudioSamplerate,miAudioFrameRate);

	Slice *s = newSlice();
	copy2Slice(s);
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mHash;
	s->misMetaData = true;
	s->mllIndex = mllIdx;
	CFlvPool::instance()->push(mHashIdx,s);
	misPushFlv = true;
	return CMS_OK;
}

int  ChttpClient::decodeVideo(char *data,int len,uint32 timestamp)
{
	if (len <= 0)
	{
		delete[] data;
		return CMS_OK;
	}
	byte vType = byte(data[0] & 0x0F);
	if (mvideoType == 0xFF || mvideoType != vType)
	{
		if (mvideoType == 0xFF)
		{
			logs->info("%s [ChttpClient::decodeVideo] http %s first video type %s",
				mremoteAddr.c_str(),murl.c_str(),getVideoType(vType).c_str());
		}
		else
		{
			logs->info("%s [ChttpClient::decodeVideo] http %s first video type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),getVideoType(mvideoType).c_str(),getVideoType(vType).c_str());
		}
		mvideoType = vType;
		misChangeMediaInfo = true;
	}
	bool isKeyFrame = false;
	FlvPoolDataType dataType = DATA_TYPE_VIDEO;
	if (vType == 0x02)
	{
		if (len <= 1)
		{
			delete[] data;
			return CMS_OK; //空帧
		}
	}
	else if (vType == 0x07)
	{
		if (len <= 5)
		{
			delete[] data;
			return CMS_OK; //空帧
		}
		if (data[0] == 0x17)
		{
			isKeyFrame = true;
			if (data[1] == 0x00)
			{
				dataType = DATA_TYPE_FIRST_VIDEO;
				miVideoFrameRate = 30;
				misChangeMediaInfo = true;
			}
			else if (data[1] == 0x01)
			{

			}
		}
	}
	Slice *s = newSlice();
	copy2Slice(s);	
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mHash;
	s->miDataType = dataType;
	s->misKeyFrame = isKeyFrame;
	s->mllIndex = mllIdx;
	if (dataType != DATA_TYPE_FIRST_VIDEO)
	{		
		mllIdx++;
		s->muiTimestamp = timestamp;
	}
	CFlvPool::instance()->push(mHashIdx,s);
	misPushFlv = true;
	return CMS_OK;
}

int  ChttpClient::decodeAudio(char *data,int len,uint32 timestamp)
{
	if (len <= 0)
	{
		delete[] data;
		return CMS_OK;//空帧
	}
	byte aType = byte(data[0]>>4 & 0x0F);
	if (maudioType == 0xFF || maudioType != aType)
	{
		if (maudioType == 0xFF)
		{
			logs->info("%s [ChttpClient::decodeAudio] http %s first audio type %s",
				mremoteAddr.c_str(),murl.c_str(),getAudioType(aType).c_str());
		}
		else
		{
			logs->info("%s [ChttpClient::decodeAudio] http %s first audio type change,old type %s,new type %s",
				mremoteAddr.c_str(),murl.c_str(),getAudioType(maudioType).c_str(),getAudioType(aType).c_str());
		}
		maudioType = aType;
		misChangeMediaInfo = true;
	}
	FlvPoolDataType dataType = DATA_TYPE_AUDIO;
	if (aType == 0x02 ||
		aType == 0x04 ||
		aType == 0x05 ||
		aType == 0x06 ){
			if (len <= 1)
			{
				delete[] data;
				return CMS_OK;//空帧
			}
	} 
	else if (aType >= 0x0A )
	{
		if (len >= 2 &&
			data[1] == 0x00)
		{
			if (len <= 2)
			{
				delete[] data;
				return CMS_OK;//空帧
			}
			dataType = DATA_TYPE_FIRST_AUDIO;
			if (len >= 4)
			{
				miAudioSamplerate = getAudioSampleRates(data);
				miAudioFrameRate = getAudioFrameRate(miAudioSamplerate);
				logs->info("%s [ChttpClient::decodeAudio] http %s audio sampleRate=%d,audio frame rate=%d",
					mremoteAddr.c_str(),murl.c_str(),miAudioSamplerate,miAudioFrameRate);
			}
		}
	}
	Slice *s = newSlice();
	copy2Slice(s);	
	s->mData = data;
	s->miDataLen = len;
	s->mhHash = mHash;
	s->miDataType = dataType;
	s->mllIndex = mllIdx;
	if (dataType != DATA_TYPE_FIRST_AUDIO)
	{
		mllIdx++;
		s->muiTimestamp = timestamp;
	}
	CFlvPool::instance()->push(mHashIdx,s);
	misPushFlv = true;
	return CMS_OK;
}

void ChttpClient::down8upBytes()
{
	unsigned long tt = getTickCount();
	if (tt - mspeedTick > 1000)
	{
		mspeedTick = tt;
		int32 bytes = mrdBuff->readBytesNum();
		if (bytes > 0 && misPushFlv)
		{
			misDown8upBytes = true;
			makeOneTaskDownload(mHash,bytes,false);
		}
		bytes = mwrBuff->writeBytesNum();
		if (bytes > 0)
		{
			makeOneTaskupload(mHash,bytes,PACKET_CONN_DATA);
		}
	}
}
