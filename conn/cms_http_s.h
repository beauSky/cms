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
#ifndef __CMS_HTTP_S_H__
#define __CMS_HTTP_S_H__
#include <interface/cms_read_write.h>
#include <interface/cms_interf_conn.h>
#include <protocol//cms_flv_transmission.h>
#include <common/cms_type.h>
#include <common/cms_binary_writer.h>
#include <protocol/cms_http.h>
#include <net/cms_net_var.h>
#include <string>

class CHttpServer:public Conn
{
public:
	CHttpServer(CReaderWriter *rw,bool isTls);
	~CHttpServer();

	int doit();
	int handleEv(FdEvents *fe);
	int stop(std::string reason);
	std::string getUrl();
	std::string getPushUrl(){return "";};
	std::string getRemoteIP();
	cms_net_ev    *evReadIO(cms_net_ev *ev = NULL);
	cms_net_ev    *evWriteIO(cms_net_ev *ev = NULL);
	void down8upBytes();

	CReaderWriter *rwConn();

	void reset();
	int  doDecode();
	int  doReadData(){return CMS_OK;};
	int  doTransmission();
	int  sendBefore(const char *data,int len);
	bool isFinish();
	int  doRead(bool isTimeout);
	int  doWrite(bool isTimeout);
private:
	int  handle();
	int  handleCrossDomain(int &ret);
	int	 handleFlv(int &ret);
	int  handleQuery(int &ret);
	int  handleM3U8(int &ret);
	int  handleTS(int &ret);
	void makeHash();
	void makeHash(std::string url);
	void tryCreateTask();
	
	cms_net_ev	*mwatcherReadIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���
	cms_net_ev	*mwatcherWriteIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���

	bool			misDecodeHeader;
	CReaderWriter	*mrw;
	std::string		murl;
	std::string		mreferer;
	std::string		mremoteAddr;
	std::string		mremoteIP;
	std::string		mlocalAddr;
	std::string		mlocalIP;
	std::string		mHost;
	HASH			mHash;
	uint32          mHashIdx;
	std::string		mstrHash;
	bool			misAddConn;		//�Ƿ������ݵ�����
	bool			misFlvRequest;
	bool			misM3U8TSRequest;
	bool			misStop;

	int64           mllIdx;
	CFlvTransmission *mflvTrans;
	CHttp			*mhttp;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;

	//�ٶ�ͳ��
	int32			mxSecdownBytes;
	int32			mxSecUpBytes;
	int32			mxSecTick;

	//websocket
	bool			misWebSocket;
	std::string		mwsOrigin               ;
	std::string		msecWebSocketAccept;
	std::string		msecWebSocketProtocol;

	BinaryWriter	*mbinaryWriter;

	unsigned long   mspeedTick;
	int64			mtimeoutTick;
};
#endif
