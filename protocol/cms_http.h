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
#ifndef __CMS_HTTP_H__
#define __CMS_HTTP_H__
#include <core/cms_buffer.h>
#include <common/cms_url.h>
#include <common/cms_var.h>
#include <common/cms_binary_writer.h>
#include <interface/cms_interf_conn.h>
#include <interface/cms_protocol.h>
#include <protocol/cms_ssl.h>
#include <string>
#include <map>

class Request
{
public:
	Request();
	~Request();

	void		setMethod(std::string method);
	bool		setUrl(std::string url);
	std::string getUrl();
	void		setHeader(std::string key,std::string value);
	std::string	getHeader(std::string key);
	std::string	getHttpParam(std::string key);
	void		clearHeader();
	std::string readRequest();
	void		setRemoteAddr(std::string addr);
	std::string getRemoteAddr();
	void		setRefer(std::string refer);
	bool		parseHeader(const char *header,int len);
	void		reset();
private:
	std::string		mremoteAddr;
	std::string		mmethod;
	LinkUrl			mlinkUrl;
	std::string		murl;
	std::string		mreferUrl;
	std::map<std::string,std::string> mmapHeader;
	std::map<std::string,std::string> mmapParam;
	std::string		mhost;
};

class Response
{
public:
	Response();
	~Response();
	std::string	getHeader(std::string key);
	void		setHeader(std::string key,std::string value);
	void		clearHeader();
	void		setUrl(std::string url);
	std::string readResponse();
	int			getStatusCode();
	void		setStatus(int statusCode,std::string statsu);
	void		setRemoteAddr(std::string addr);
	bool		parseHeader(const char *header,int len);
	std::string getResponse();
	void		reset();
private:
	std::string		moriUrl;
	std::string		mremoteAddr;
	std::string		mstatus;		//"200 OK"
	int				mstatusCode;	//200
	std::string		mproto;			//"HTTP/1.0"
	std::string     moriRsp;
	std::map<std::string,std::string> mmapHeader;
};

class CHttp: public CProtocol
{
public:
	CHttp(Conn *super,CBufferReader *rd,
		CBufferWriter *wr,CReaderWriter *rw,std::string remoteAddr,
		bool isClient,bool isTls);
	~CHttp();

	bool        run();
	int			want2Read(bool isTimeout);
	int			want2Write(bool isTimeout);
	int         read(char **data,int &len);			//���ݱ�����*data�У���ȡָ�����ȵ�����len�����û���㹻�����򷵻�0
													//������֮��Ҫ���ϴ���û����ǰ�����ٴ�read��������ܳ��ֲ���Ԥ��Ĵ���
	int         write(const char *data,int &len);
	bool		setUrl(std::string url);
	void		setRefer(std::string refer);
	void		setOriUrl(std::string oriUrl);
	Request		*httpRequest();
	Response	*httpResponse();
	//�����麯��
	int sendMetaData(Slice *s);
	int sendVideoOrAudio(Slice *s,uint32 uiTimestamp);
	int writeBuffSize();
	void setWriteBuffer(int size);
	std::string remoteAddr();
	std::string getUrl();
	void syncIO();
	bool isCmsConnection();
	std::string protocol();

	void setChunked();
	cms_timer *cmsTimer2Write();
	cms_timer *cmsTimer2Read();
private:
	void doWriteTimeout();
	void doReadTimeout();
	int  readChunkedRN();	
	void reset();

	//tls �Ķ���
	bool			misTls;
	CSSL			*mssl;

	Request			*mrequest; //��Ϊ�ͻ������󣬻��߱������������Ϣ
	Response		*mresponse;//��Ϊ�ͻ��˵���Ӧ��������Ӧ���������Ϣ
	bool			misReadHeader;
	bool			misWriteHeader;
	int				mheaderEnd;
	std::string     mstrHeader;

	bool			misChunked;
	int64			mchunkedLen;
	bool			misReadChunkedLen;
	std::string		mchunkBytesRN;
	int				mchunkedReadrRN;

	int				msendRequestLen;
	bool			misReadReuqest;
	std::string		mstrRequestHeader;

	Conn			*msuper;
	bool			misClient;
	int				mcmsReadTimeOutDo;
	int				mcmsWriteTimeOutDo;	
	std::string		mreferUrl;
	std::string		moriUrl;

	std::string		mremoteAddr;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;
	CReaderWriter	*mrw;
	CByteReaderWriter *mbyteReadWrite;

	cms_timer		*mcmsReadTimeout;
	cms_timer		*mcmsWriteTimeout;
	BinaryWriter	*mbinaryWriter;

	std::string		msProtocol;
	int64			mwebsocketLen;
	bool			mwebsocketIsMask;
	int				mwebsocketMask[4];
};
#endif
