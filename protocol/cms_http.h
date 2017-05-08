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
	void		clearHeader();
	std::string readRequest();
	void		setRemoteAddr(std::string addr);
	std::string getRemoteAddr();
	void		setRefer(std::string refer);
	bool		parseHeader(const char *header,int len);
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
	int			want2Read();
	int			want2Write(bool isTimeout);
	int         read(char **data,int &len);			//数据保存在*data中，读取指定长度的数据len，如果没有足够数据则返回0
													//，读完之后要马上处理，没处理前不能再次read，否则可能出现不可预测的错误
	int         write(const char *data,int &len);
	bool		setUrl(std::string url);
	void		setRefer(std::string refer);
	void		setOriUrl(std::string oriUrl);
	Request		*httpRequest();
	Response	*httpResponse();
	//重载虚函数
	int sendMetaData(Slice *s);
	int sendVideoOrAudio(Slice *s,uint32 uiTimestamp);
	int writeBuffSize();
	std::string remoteAddr();
	std::string getUrl();
	void syncIO();
	void setChunked();
private:
	void		doWriteTimeout();
	int 		readChunkedRN();		
	//tls 的东西
	bool			misTls;
	CSSL			*mssl;

	Request			*mrequest; //作为客户端请求，或者被请求的请求信息
	Response		*mresponse;//作为客户端的响应，或者响应被请求的信息
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
};
#endif
