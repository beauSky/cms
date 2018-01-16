/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: Ìì¿ÕÃ»ÓÐÎÚÔÆ/kisslovecsh@foxmail.com

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
#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <log/cms_log.h>
#include <vector>
#include <string>

class CAddr
{
public:
	CAddr(char *addr,int defaultPort);
	~CAddr();
	char *addr();
	char *host();
	int  port();
private:
	char	*mAddr;
	char    *mHost;
	int		mPort;
};

class Clog
{
public:
	Clog(char *path,char *level,bool console,int size);
	~Clog();
	char		*path();
	int			size();
	LogLevel	level();
	bool		console();
private:
	char		*mpath;
	int			msize;
	bool		mconsole;
	LogLevel	mlevel;
};

class CertKey
{
public:
	CertKey(char *cert,char *key,char *dhparam,char *ciper);
	CertKey();
	~CertKey();

	char *certificateChain();
	char *privateKey();
	char *cipherPrefs();
	char *dhparam();
	bool isOpenSSL();
private:
	bool misOpen;
	char *mcert;
	char *mkey;
	char *mcipher;
	char *mdhparam;
};

class CUpperAddr
{
public:
	CUpperAddr();
	~CUpperAddr();

	void		addPull(std::string addr);
	std::string getPull(unsigned int i);
	void		addPush(std::string addr);
	std::string getPush(unsigned int i);	
private:
	std::vector<std::string> mvPullAddr;
	std::vector<std::string> mvPushAddr;
	
};

class CUdpFlag
{
public:
	CUdpFlag();
	~CUdpFlag();
	bool		isOpenUdpPull();
	bool		isOpenUdpPush();
	void		setUdp(bool isPull,bool isPush);
	int			udpConnNum();
private:
	bool	    misOpenUdpPull;
	bool	    misOpenUdpPush;
	int			miUdpMaxConnNum;
};

class CConfig
{
public:
	CConfig();
	~CConfig();
	static CConfig *instance();
	static void freeInstance();
	bool		init(const char *configPath);
	CAddr		*addrHttp();
	CAddr		*addrHttps();
	CAddr		*addrRtmp();
	CAddr		*addrQuery();
	CertKey		*certKey();
	Clog		*clog();
	CUpperAddr	*upperAddr();
	CUdpFlag	*udpFlag();
private:
	static CConfig *minstance;

	CAddr	   *mHttp;
	CAddr	   *mHttps;
	CAddr	   *mRtmp;
	CAddr	   *mQuery;
	CUpperAddr *muaAddr;
	Clog	   *mlog;
	CertKey    *mcertKey;
	CUdpFlag   *mUdpFlag;
};
#endif
