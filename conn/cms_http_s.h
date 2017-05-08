#ifndef __CMS_HTTP_S_H__
#define __CMS_HTTP_S_H__
#include <interface/cms_read_write.h>
#include <interface/cms_interf_conn.h>
#include <protocol//cms_flv_transmission.h>
#include <common/cms_type.h>
#include <common/cms_binary_writer.h>
#include <protocol/cms_http.h>
#include <libev/ev.h>
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
	struct ev_loop  *evLoop();
	struct ev_io    *evReadIO();
	struct ev_io    *evWriteIO();
	void down8upBytes();

	void setEVLoop(struct ev_loop *loop);

	int doDecode();
	int doReadData(){return CMS_OK;};
	int doTransmission();
	int sendBefore(const char *data,int len);

	int doRead();
	int doWrite(bool isTimeout);
private:
	int  handle();
	int	 handleFlv(int &ret);
	int handleQuery(int &ret);
	void makeHash();
	void tryCreateTask();
	
	struct ev_loop	*mloop;			//ȫ�ֲ����ڱ���
	struct ev_io	*mwatcherReadIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���
	struct ev_io	*mwatcherWriteIO;	//��Ȼ�����洴�� cms_conn_mgr ���� cms_net_dispatch �����������ڱ���

	bool			misDecodeHeader;
	CReaderWriter	*mrw;
	std::string		murl;
	std::string		mreferer;
	std::string		mremoteAddr;
	std::string		mremoteIP;
	std::string		mHost;
	HASH			mHash;
	uint32          mHashIdx;
	std::string		mstrHash;
	bool			misAddConn;		//�Ƿ������ݵ�����
	bool			misFlvRequest;
	bool			misStop;

	int64           mllIdx;
	CFlvTransmission *mflvTrans;
	CHttp			*mhttp;
	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;

	//websocket
	bool			misWebSocket;
	std::string		mwsOrigin               ;
	std::string		msecWebSocketAccept;
	std::string		msecWebSocketProtocol;

	BinaryWriter	*mbinaryWriter;

	unsigned long  mspeedTick;
};
#endif
