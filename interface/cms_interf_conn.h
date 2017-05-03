#ifndef __CMS_INTERFACE_CONN_H__
#define __CMS_INTERFACE_CONN_H__
#include <string>
#include <common/cms_var.h>
#include <libev/ev.h>

class Conn
{
public:
	Conn();
	virtual ~Conn();
	virtual int doit() = 0;
	virtual int handleEv(FdEvents *fe) = 0;
	virtual int stop(std::string reason) = 0;
	virtual std::string getUrl() = 0;
	virtual std::string getPushUrl() = 0;
	virtual std::string getRemoteIP() = 0;

	virtual struct ev_loop  *evLoop() = 0;
	virtual struct ev_io    *evReadIO() = 0;
	virtual struct ev_io    *evWriteIO() = 0;
	//http สนำร
	virtual int doDecode() = 0;
	virtual int doTransmission() = 0;
	virtual int sendBefore(const char *data,int len) = 0;
};

#endif
