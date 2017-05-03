#ifndef __CMS_SSL_H__
#define __CMS_SSL_H__
#include <core/cms_buffer.h>
#include <s2n/s2n.h>
#include <string>

class CSSL
{
public:
	CSSL(int fd,std::string remoteAddr,bool isClient);
	~CSSL();
	bool	run();
	int		read(char **data,int &len);
	int		write(const char *data,int &len);
	int		bufferWriteSize();
	bool    isHandShake();
	int		flush();
	bool	isUsable();
private:
	int		handShakeTLS();
	bool					misTlsHandShake;
	struct s2n_connection	*ms2nConn;
	struct s2n_config		*mconfig;
	bool					misClient;
	int						mfd;
	std::string				mremoteAddr;

	CBufferReader	*mrdBuff;
	CBufferWriter	*mwrBuff;

	//·¢ËÍ»º´æ
	char *mbuffer;
	int  mb;
	int  me;
	int  mbufferSize;
};
#endif
