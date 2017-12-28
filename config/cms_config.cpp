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
#include <config/cms_config.h>
#include <cJSON/cJSON.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
using namespace std;


CAddr::CAddr(char *addr,int defaultPort)
{	
	assert(strlen(addr) < 128);
	mAddr = new char[128];
	memset(mAddr,0,128);
	mHost = new char[128];
	memset(mHost,0,128);
	strcpy(mAddr,addr);
	char *p = strstr(addr,":");
	if (p == NULL)
	{
		strcpy(mHost,addr);
		mPort = defaultPort;
	}
	else
	{
		memcpy(mHost,addr,p-addr);
		mPort = atoi(p+1);
	}
	printf("### CAddr addr=%s,defaultPort=%d,host=%s,port=%d ###\n",mAddr,defaultPort,mHost,mPort);
}

CAddr::~CAddr()
{
	if (mAddr)
	{
		delete[] mAddr;
		mAddr = NULL;
	}
	if (mHost)
	{
		delete[] mHost;
		mAddr = NULL;
	}
}

char *CAddr::addr()
{
	printf("### CAddr get addr=%s ###\n",mAddr);
	return mAddr;
}

char *CAddr::host()
{
	return mHost;
}

int  CAddr::port()
{
	return mPort;
}

LogLevel getLevel(string level) 
{
	transform(level.begin(), level.end(), level.begin(), (int (*)(int))tolower);
	if (level == "off")
	{
		return OFF;
	}
	else if (level == "debug")
	{
		return DEBUG;
	}
	else if (level == "info")
	{
		return INFO;
	}
	else if (level == "warn")
	{
		return WARN;
	}
	else if (level == "error")
	{
		return ERROR1;
	}
	else if (level == "fatal")
	{
		return FATAL;
	}
	else if (level == "all_level")
	{
		return ALL_LEVEL;
	}
	return DEBUG;
}

Clog::Clog(char *path,char *level,bool console,int size)
{
	assert(path!=NULL);
	int len = strlen(path);
	mpath = new char[len+1];
	memcpy(mpath,path,len);
	mpath[len] = '\0';
	if (size < 500*1024*1024)
	{
		msize = 500*1024*1024;
	}
	msize = size;
	mlevel = getLevel(level);
	mconsole = console;
}

Clog::~Clog()
{
	if (mpath)
	{
		delete[] mpath;
		mpath = NULL;
	}
}

char *Clog::path()
{
	return mpath;
}

int	Clog::size()
{
	return msize;
}

LogLevel Clog::level()
{
	return mlevel;
}

bool Clog::console()
{
	return mconsole;
}

CertKey::CertKey(char *cert,char *key,char *dhparam,char *ciper)
{
	assert(cert!=NULL);
	int len = strlen(cert);
	mcert = new char[len+1];
	memcpy(mcert,cert,len);
	mcert[len] = '\0';

	assert(key!=NULL);
	len = strlen(key);
	mkey = new char[len+1];
	memcpy(mkey,key,len);
	mkey[len] = '\0';
	
	assert(ciper!=NULL);
	len = strlen(ciper);
	mcipher = new char[len+1];
	memcpy(mcipher,ciper,len);
	mcipher[len] = '\0';

	assert(dhparam!=NULL);
	len = strlen(dhparam);
	mdhparam = new char[len+1];
	memcpy(mdhparam,dhparam,len);
	mdhparam[len] = '\0';

	misOpen = true;
}

CertKey::CertKey()
{
	misOpen = false;
	mcert = NULL;
	mkey = NULL;
	mcipher = NULL;
}

CertKey::~CertKey()
{
	if (mcert)
	{
		delete[] mcert;
	}
	if (mkey)
	{
		delete[] mkey;
	}
	if (mcipher)
	{
		delete[] mcipher;
	}
}

char *CertKey::certificateChain()
{
	return mcert;
}

char *CertKey::privateKey()
{
	return mkey;
}

char *CertKey::cipherPrefs()
{
	return mcipher;
}

char *CertKey::dhparam()
{
	return mdhparam;
}

bool CertKey::isOpenSSL()
{
	return misOpen;
}


CUpperAddr::CUpperAddr()
{

}

CUpperAddr::~CUpperAddr()
{

}

void CUpperAddr::addPull(std::string addr)
{
	mvPullAddr.push_back(addr);
}

std::string CUpperAddr::getPull(unsigned int i)
{
	if (mvPullAddr.empty())
	{
		return "";
	}
	i = i % mvPullAddr.size();
	return mvPullAddr.at(i);
}

void CUpperAddr::addPush(std::string addr)
{
	mvPushAddr.push_back(addr);
}

std::string CUpperAddr::getPush(unsigned int i)
{
	if (mvPushAddr.empty())
	{
		return "";
	}
	i = i % mvPushAddr.size();
	return mvPushAddr.at(i);
}


CUdpFlag::CUdpFlag()
{
	misOpenUdpPull = false;
	misOpenUdpPush = false;
	miUdpMaxConnNum = 5000000;
}

CUdpFlag::~CUdpFlag()
{

}

bool CUdpFlag::isOpenUdpPull()
{
	return misOpenUdpPull;
}

bool CUdpFlag::isOpenUdpPush()
{
	return misOpenUdpPush;
}

void CUdpFlag::setUdp(bool isPull,bool isPush)
{
	misOpenUdpPull = isPull;
	misOpenUdpPush = isPush;
}

int CUdpFlag::udpConnNum()
{
	if (miUdpMaxConnNum < 300000)
	{
		miUdpMaxConnNum = 300000;
	}
	return miUdpMaxConnNum;
}

CConfig *CConfig::minstance = NULL;
CConfig *CConfig::instance()
{
	if (minstance == NULL)
	{
		minstance = new CConfig();
	}
	return minstance;
}

void CConfig::freeInstance()
{
	if (minstance)
	{
		delete minstance;
		minstance = NULL;
	}
}

CConfig::CConfig()
{
	mHttp = NULL;
	mHttps = NULL;
	mRtmp = NULL;
	mQuery = NULL;
	muaAddr = NULL;
	mlog = NULL;
	mcertKey = NULL;
	mUdpFlag = NULL;
}

CConfig::~CConfig()
{
	if (mHttp)
	{
		delete mHttp;
	}
	if (mHttps)
	{
		delete mHttps;
	}
	if (mRtmp)
	{
		delete mRtmp;
	}
	if (mQuery)
	{
		delete mQuery;
	}
	if (mlog)
	{
		delete mlog;
	}
	if (mcertKey)
	{
		delete mcertKey;
	}
	if (muaAddr)
	{
		delete muaAddr;
	}
	if (mUdpFlag)
	{
		delete mUdpFlag;
	}
}

bool	CConfig::init(const char *configPath)
{
	FILE *fp = fopen(configPath,"rb");
	if (fp == NULL)
	{
		printf("*** [CConfig::init] open file %s fail,errno=%d,strerrno=%s ***\n",
			configPath,errno,strerror(errno));
		return false;
	}
	fseek(fp,0,SEEK_END);
	long len = ftell(fp);
	if (len <= 0)
	{
		printf("*** [CConfig::init] file %s is empty=%ld ***\n",
			configPath, len);
		fclose(fp);
		return false;
	}
	char *data = new char[len +1];
	fseek(fp,0,SEEK_SET);
	int n = fread(data,1, len,fp);
	if (n != len)
	{
		printf("*** [CConfig::init] fread file %s fail,errno=%d,strerrno=%s ***\n",
			configPath,errno,strerror(errno));
		fclose(fp);
		return false;
	}
	data[len] = '\0';
	fclose(fp);

	cJSON *root = cJSON_Parse(data);
	if (root == NULL)
	{
		printf("*** [CConfig::init] config file %s parse json fail %s ***\n",
			configPath, data);
		delete[] data;
		return false;
	}
	
	delete[] data;
	data = NULL;
	//监听端口
	cJSON *listenObject = cJSON_GetObjectItem(root, "listen");
	if (listenObject == NULL)
	{
		printf("*** [CConfig::init] config file %s do not have [listen] ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	string httpListen, httpsListen, rtmpListen, queryListen;
	cJSON *T = cJSON_GetObjectItem(listenObject, "http");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s listen term do not have http ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	httpListen = T->valuestring;
	T = cJSON_GetObjectItem(listenObject, "https");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s listen term do not have https ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	httpsListen = T->valuestring;
	T = cJSON_GetObjectItem(listenObject, "rtmp");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s listen term do not have rtmp ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	rtmpListen = T->valuestring;
	T = cJSON_GetObjectItem(listenObject, "query");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s listen term do not have query ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	queryListen = T->valuestring;
	
	mHttp = new CAddr((char *)httpListen.c_str(),80);
	mHttps = new CAddr((char *)httpsListen.c_str(),443);
	mRtmp = new CAddr((char *)rtmpListen.c_str(),1935);
	mQuery = new CAddr((char *)queryListen.c_str(),8981);

	muaAddr = new CUpperAddr;
	//上层节点
	cJSON *upper = cJSON_GetObjectItem(root, "upper");
	if (upper != NULL)
	{
		//pull addr
		cJSON *pull = cJSON_GetObjectItem(upper, "pull");
		if (pull != NULL)
		{
			if (pull->type == cJSON_Array)
			{
				int iSize = cJSON_GetArraySize(pull);
				for (int i = 0; i < iSize; i++)
				{
					cJSON *T = cJSON_GetArrayItem(pull, i);
					if (T->type != cJSON_String)
					{
						printf("***** upper pull addr is not string *****\n");
						cJSON_Delete(root);
						return false;
					}
					muaAddr->addPull(T->valuestring);
				}
			}
			else
			{
				printf("***** upper pull config is not array *****\n");
				cJSON_Delete(root);
				return false;
			}
		}

		//push addr
		cJSON *push = cJSON_GetObjectItem(upper, "push");
		if (push != NULL)
		{
			if (push->type == cJSON_Array)
			{
				int iSize = cJSON_GetArraySize(push);
				for (int i = 0; i < iSize; i++)
				{
					cJSON *T = cJSON_GetArrayItem(push, i);
					if (T->type != cJSON_String)
					{
						printf("***** upper push addr is not string *****\n");
						cJSON_Delete(root);
						return false;
					}
					muaAddr->addPush(T->valuestring);
				}
			}
			else
			{
				printf("***** upper push config is not array *****\n");
				cJSON_Delete(root);
				return false;
			}
		}
	}
	//udp 属性
	mUdpFlag = new CUdpFlag;
	cJSON *udp = cJSON_GetObjectItem(root, "udp");
	if (udp != NULL &&
		udp->type == cJSON_Object)
	{
		bool isOpenPull = false;
		bool isOpenPush = false;
		//pull flag
		cJSON *T = cJSON_GetObjectItem(udp, "open_pull");
		if (T != NULL &&
			(T->type == cJSON_True || T->type == cJSON_False))
		{
			if (T->type == cJSON_True)
			{
				isOpenPull = true;
			}
		}
		else
		{
			printf("***** udp open_pull config is not bool *****\n");
			cJSON_Delete(root);
			return false;
		}

		//push flag
		T = cJSON_GetObjectItem(udp, "open_push");
		if (T != NULL &&
			(T->type == cJSON_True || T->type == cJSON_False))
		{
			if (T->type == cJSON_True)
			{
				isOpenPush = true;
			}			
		}
		else
		{
			printf("***** udp open_push config is not bool *****\n");
			cJSON_Delete(root);
			return false;
		}

		mUdpFlag->setUdp(isOpenPull,isOpenPush);
	}

	//证书
	cJSON *tls = cJSON_GetObjectItem(root, "tls");
	if (tls != NULL
		&& tls->type == cJSON_Object)
	{
		string cert, key, dhparam, cipher;
		T = cJSON_GetObjectItem(tls, "cert");
		if (T == NULL || T->type != cJSON_String)
		{
			printf("*** [CConfig::init] config file %s tls term do not have cert ***\n",
				configPath);
			cJSON_Delete(root);
			return false;
		}
		cert = T->valuestring;

		T = cJSON_GetObjectItem(tls, "key");
		if (T == NULL || T->type != cJSON_String)
		{
			printf("*** [CConfig::init] config file %s tls term do not have key ***\n",
				configPath);
			cJSON_Delete(root);
			return false;
		}
		key = T->valuestring;

		T = cJSON_GetObjectItem(tls, "dhparam");
		if (T == NULL || T->type != cJSON_String)
		{
			printf("*** [CConfig::init] config file %s tls term do not have dhparam ***\n",
				configPath);
			cJSON_Delete(root);
			return false;
		}
		dhparam = T->valuestring;

		T = cJSON_GetObjectItem(tls, "cipher");
		if (T == NULL || T->type != cJSON_String)
		{
			printf("*** [CConfig::init] config file %s tls term do not have cipher ***\n",
				configPath);
			cJSON_Delete(root);
			return false;
		}
		cipher = T->valuestring;
		
		FILE *fp = fopen(cert.c_str(),"rb");
		if (fp == NULL)
		{
			printf("*** [CConfig::init] open cert file %s fail,errstr=%s ***\n",
				cert.c_str(),strerror(errno));
			cJSON_Delete(root);
			return false;
		}
		fseek(fp,0,SEEK_END);
		int len = ftell(fp);
		fseek(fp,0,SEEK_SET);
		char *pcert = new char[len+1];
		fread(pcert,1,len,fp);
		pcert[len] = '\0';
		fclose(fp);		

		fp = fopen(key.c_str(),"rb");
		if (fp == NULL)
		{
			printf("*** [CConfig::init] open key file %s fail,errstr=%s ***\n",
				key.c_str(),strerror(errno));
			delete[] pcert;
			cJSON_Delete(root);
			return false;
		}
		fseek(fp,0,SEEK_END);
		len = ftell(fp);
		fseek(fp,0,SEEK_SET);
		char *pkey = new char[len+1];
		fread(pkey,1,len,fp);
		pkey[len] = '\0';
		fclose(fp);

		fp = fopen(dhparam.c_str(),"rb");
		if (fp == NULL)
		{
			printf("*** [CConfig::init] open dhparam file %s fail,errstr=%s ***\n",
				dhparam.c_str(),strerror(errno));
			delete[] pcert;
			cJSON_Delete(root);
			return false;
		}
		fseek(fp,0,SEEK_END);
		len = ftell(fp);
		fseek(fp,0,SEEK_SET);
		char *pdhparam = new char[len+1];
		fread(pdhparam,1,len,fp);
		pdhparam[len] = '\0';
		fclose(fp);

		mcertKey = new CertKey(pcert,
			pkey,
			pdhparam,
			(char *)cipher.c_str());

		delete[] pcert;
		delete[] pkey;
		delete[] pdhparam;
	}
	else
	{
		mcertKey = new CertKey();
	}

	//日志
	cJSON* log = cJSON_GetObjectItem(root, "log");
	if (log == NULL || log->type != cJSON_Object)
	{
		printf("*** [CConfig::init] config file %s do not have [log] ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	string file, level;
	int size = 0;
	T = cJSON_GetObjectItem(log, "file");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s log term do no have file ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	file = T->valuestring;

	T = cJSON_GetObjectItem(log, "size");
	if (T == NULL || T->type != cJSON_Number)
	{
		printf("*** [CConfig::init] config file %s log term do no have size ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	size = T->valueint;

	T = cJSON_GetObjectItem(log, "level");
	if (T == NULL || T->type != cJSON_String)
	{
		printf("*** [CConfig::init] config file %s log term do no have level ***\n",
			configPath);
		cJSON_Delete(root);
		return false;
	}
	level = T->valuestring;

	bool console = false;
	T = cJSON_GetObjectItem(log, "console");
	if (T == NULL || T->type == cJSON_True)
	{
		console = true;
	}
	
	mlog = new Clog((char *)file.c_str(),
		(char *)level.c_str(),
		console,
		size);
	return true;
}

CAddr *CConfig::addrHttp()
{
	return mHttp;
}

CAddr *CConfig::addrHttps()
{
	return mHttps;
}

CAddr *CConfig::addrRtmp()
{
	return mRtmp;
}

CAddr *CConfig::addrQuery()
{
	return mQuery;
}

CertKey *CConfig::certKey()
{
	return mcertKey;
}

Clog *CConfig::clog()
{
	return mlog;
}

CUpperAddr *CConfig::upperAddr()
{
	return muaAddr;
}

CUdpFlag *CConfig::udpFlag()
{
	return mUdpFlag;
}
