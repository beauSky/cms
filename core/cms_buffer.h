#ifndef __CMS_BUFFER_H__
#define __CMS_BUFFER_H__
#include <interface/cms_read_write.h>
#include <common/cms_type.h>
#include <s2n/s2n.h>

class CBufferReader
{
public:
	CBufferReader(CReaderWriter *rd,int size);
	CBufferReader(s2n_connection *s2nConn,int size);
	~CBufferReader();
	char  *readBytes(int n);
	char  *peek(int n);
	char  readByte(); 
	void  skip(int n);
	int   size();
	int   grow(int n);
	char  *errnoCode();
	int   errnos();
	void  close();
	int32 readBytesNum();
private:
	void  resize();
	int   seterrno(int err);
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
	int  merrcode;
	int32 mtotalReadBytes;
	CReaderWriter *mrd;
	struct s2n_connection	*ms2nConn;
};

class CBufferWriter
{
public:
	CBufferWriter(CReaderWriter *rd,int size = 128*1024);
	CBufferWriter(s2n_connection *s2nConn,int size = 128*1024);
	~CBufferWriter();
	int writeBytes(const char *data,int n); //要么出错返回 -1,要么返回实际发送的数据,如果本次没发送完毕，会保存到底层
	int writeByte(char ch);
	int flush();
	bool  isUsable();
	void  resize();
	char  *errnoCode();
	int   errnos();
	int   size();
	void  close();
	int32 writeBytesNum();
private:
	int   seterrno(int err);
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
	int  merrcode;
	int32 mtotalWriteBytes;
	CReaderWriter *mrd;
	struct s2n_connection	*ms2nConn;
};

class CByteReaderWriter
{
public:
	CByteReaderWriter(int size = 128*1024);
	~CByteReaderWriter();
	int	  writeBytes(const char *data,int n);
	int   writeByte(char ch);
	char  *readBytes(int n);
	char  readByte(); 
	char  *peek(int n);
	void  skip(int n);
	void  resize();
	int   size();
private:
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
};
#endif
