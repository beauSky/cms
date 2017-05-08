#include <protocol/cms_ssl.h>
#include <log/cms_log.h>
#include <common/cms_utility.h>
#include <config/cms_config.h>

CSSL::CSSL(int fd,std::string remoteAddr,bool isClient)
{
	ms2nConn = NULL;
	mconfig = NULL;
	misTlsHandShake = false;
	misClient = isClient;
	mfd = fd;
	mremoteAddr = remoteAddr;
	mrdBuff = NULL;
	mwrBuff = NULL;
}

CSSL::~CSSL()
{
	if (ms2nConn)
	{
		s2n_connection_free(ms2nConn);
		ms2nConn = NULL;
	}
	if (mconfig)
	{
		if (s2n_config_free(mconfig) < 0) 
		{
			logs->error("***** %s [CSSL::CSSL] error freeing configuration: '%s' *****",
				mremoteAddr.c_str(), s2n_strerror(s2n_errno, "EN"));
		}
	}
}

bool CSSL::run()
{
	if (misClient)
	{
		ms2nConn = s2n_connection_new(S2N_CLIENT);
		mconfig = s2n_config_new();
		if (mconfig == NULL) {
			logs->error("***** %s [CSSL::run] error getting new config: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}

		if (s2n_config_set_status_request_type(mconfig, S2N_STATUS_REQUEST_OCSP) < 0) {
			logs->error("***** %s [CSSL::run] error setting status request type: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (s2n_connection_set_config(ms2nConn, mconfig) < 0) 
		{
			logs->error("***** %s [CSSL::run] error client setting configuration: '%s' *****", 
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (s2n_set_server_name(ms2nConn, "cms server") < 0) 
		{
			logs->error("***** %s [CSSL::run] error setting server name: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
	}
	else
	{
		ms2nConn = s2n_connection_new(S2N_SERVER);
		mconfig = s2n_config_new();
		if (mconfig == NULL) {
			logs->error("***** %s [CSSL::run] 2 error getting new config: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (s2n_config_add_cert_chain_and_key(mconfig, 
			CConfig::instance()->certKey()->certificateChain(),
			CConfig::instance()->certKey()->privateKey()) < 0) 
		{
			logs->error("***** %s [CSSL::run] set certificate/key fail: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (CConfig::instance()->certKey()->dhparam() != NULL &&
			s2n_config_add_dhparams(mconfig, CConfig::instance()->certKey()->dhparam()) < 0) 
		{
			logs->error("***** %s [CSSL::run] adding DH parameters fail: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (CConfig::instance()->certKey()->cipherPrefs() != NULL &&
			s2n_config_set_cipher_preferences(mconfig, CConfig::instance()->certKey()->cipherPrefs()) < 0) 
		{
			logs->error("***** %s [CSSL::run] setting cipher prefs fail: '%s' *****",
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
		if (s2n_connection_set_config(ms2nConn, mconfig) < 0) 
		{
			logs->error("***** %s [CSSL::run] error server setting configuration: '%s' *****", 
				mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
			return false;
		}
	}
	if (s2n_connection_set_fd(ms2nConn, mfd) < 0) 
	{
		logs->error("***** %s [CSSL::run] error setting file descriptor: '%s' *****", 
			mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"));
		return false;
	}
	return true;
}

bool CSSL::isHandShake()
{
	return misTlsHandShake;
}

int CSSL::read(char **data,int &len)
{
	if (!misClient && !misTlsHandShake)
	{
		int ret = handShakeTLS();
		if (ret == -1)
		{
			return -1;
		}
		if (ret == 0)
		{
			return 0;
		}
	}
	if ( mrdBuff->size() < len && mrdBuff->grow(len) == CMS_ERROR)
	{
		return -1;
	}
	if (mrdBuff->size() < len)
	{
		return 0;
	}
	*data = mrdBuff->readBytes(len);
	return len;
}

int	CSSL::peek(char **data,int &len)
{
	if (!misClient && !misTlsHandShake)
	{
		int ret = handShakeTLS();
		if (ret == -1)
		{
			return -1;
		}
		if (ret == 0)
		{
			return 0;
		}
	}
	if ( mrdBuff->size() < len && mrdBuff->grow(len) == CMS_ERROR)
	{
		return -1;
	}
	if (mrdBuff->size() < len)
	{
		return 0;
	}
	*data = mrdBuff->peek(len);
	return len;
}

void CSSL::skip(int len)
{
	mrdBuff->skip(len);
}

int CSSL::write(const char *data,int &len)
{	
	if (misClient && !misTlsHandShake)
	{
		int ret = handShakeTLS();
		if (ret == -1)
		{
			return -1;
		}
		if (ret == 0)
		{
			return 0;
		}
		if (data == NULL)
		{
			return 1;
		}
	}
	else
	{
		if (mwrBuff->writeBytes(data,len) == CMS_ERROR)
		{
			logs->error("***** %s [CSSL::write] write fail: '%s' *****", 
				mremoteAddr.c_str(),mwrBuff->errnoCode());
			return -1;
		}
	}
	return len;
}

int CSSL::flush()
{
	int ret = mwrBuff->flush();
	if (ret == CMS_ERROR)
	{
		logs->error("%s [CSSL::flush] flush fail,errno=%d,strerrno=%s ***",
			mremoteAddr.c_str(),mwrBuff->errnos(),mwrBuff->errnoCode());
		return CMS_ERROR;
	}	
	return CMS_OK;
}

bool CSSL::isUsable()
{
	return mwrBuff->isUsable();
}

int CSSL::bufferWriteSize()
{
	return mwrBuff->size();
}

int CSSL::handShakeTLS()
{
	s2n_blocked_status blocked;
	if (s2n_negotiate(ms2nConn,&blocked) < 0)
	{
		logs->error("***** %s [CSSL::handShakeTLS] failed to negotiate: '%s' %d *****", 
			mremoteAddr.c_str(),s2n_strerror(s2n_errno, "EN"), s2n_connection_get_alert(ms2nConn));
		return -1;
	}
	if (blocked == S2N_NOT_BLOCKED)
	{
		logs->debug("%s [CSSL::handShakeTLS] TLS handshake succ ", 
			mremoteAddr.c_str());
		misTlsHandShake = true;

		mrdBuff = new CBufferReader(ms2nConn,128*1024);
		mwrBuff = new CBufferWriter(ms2nConn);

		return 1;
	}
	return 0;
}