#ifndef __CMS_BUFFER_H__
#define __CMS_BUFFER_H__
#include <interface/cms_read_write.h>
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
private:
	void  resize();
	int   seterrno(int err);
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
	int  merrcode;
	CReaderWriter *mrd;
	struct s2n_connection	*ms2nConn;
};

class CBufferWriter
{
public:
	CBufferWriter(CReaderWriter *rd,int size = 128*1024);
	CBufferWriter(s2n_connection *s2nConn,int size = 128*1024);
	~CBufferWriter();
	int writeBytes(const char *data,int n); //Ҫô������ -1,Ҫô����ʵ�ʷ��͵�����,�������û������ϣ��ᱣ�浽�ײ�
	int writeByte(char ch);
	int flush();
	bool  isUsable();
	void  resize();
	char  *errnoCode();
	int   errnos();
	int   size();
	void  close();
private:
	int   seterrno(int err);
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
	int  merrcode;
	CReaderWriter *mrd;
	struct s2n_connection	*ms2nConn;
};
#endif
