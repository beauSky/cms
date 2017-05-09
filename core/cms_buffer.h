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
	int writeBytes(const char *data,int n); //Ҫô������ -1,Ҫô����ʵ�ʷ��͵�����,�������û������ϣ��ᱣ�浽�ײ�
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
