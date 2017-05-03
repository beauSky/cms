#ifndef __CMS_INTERFACE_PROTOCOL_H__
#define __CMS_INTERFACE_PROTOCOL_H__
#include <flvPool/cms_flv_pool.h>
#include <string>

class CProtocol
{
public:
	CProtocol();
	virtual ~CProtocol();
	virtual int sendMetaData(Slice *s) = 0;
	virtual int sendVideoOrAudio(Slice *s,uint32 uiTimestamp) = 0;
	virtual int writeBuffSize() = 0;
	virtual void syncIO() = 0;
	virtual std::string remoteAddr() = 0;
	virtual std::string getUrl() = 0;
};
#endif
