#ifndef __CMS_READ_WRITE_H__
#define __CMS_READ_WRITE_H__
#include <string>

class CReaderWriter 
{
public:
	CReaderWriter();
	virtual ~CReaderWriter();
	virtual int   read(char *dstBuf,int len,int &nread) = 0;
	virtual int   write(char *srcBuf,int len,int &nwrite) = 0;
	virtual int   remoteAddr(char *addr,int len) = 0;
	virtual int   fd() = 0;
	virtual void  close() = 0;
	virtual char  *errnoCode() = 0;
	virtual int   errnos() = 0;
	virtual int   setNodelay(int on) = 0; 
};

#endif