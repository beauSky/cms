#include <stdio.h>
#include <common/cms_utility.h>
#include <common/cms_url.h>
#include <enc/cms_sha1.h>
#include <enc/cms_base64.h>
#include <log/cms_log.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <algorithm>

using namespace std;

void urlEncode(const char *src, int nLenSrc, char *dest, int& nLenDest)
{
	static char unreserved[256] = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, /* 0123456789 */
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, /* ABCDEFGHIJKLMNO */
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, /* PQRSTUVWXYZ*/
			0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, /* abcdefghijklmnop */
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, /* qrstuvwxyz */
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		nLenDest = 0;
		for (int i=0;  i<nLenSrc; src++,i++)
		{   
			if (unreserved[(unsigned char) *src])
			{
				*dest++ = *src;
				nLenDest++;
			}
			else
			{
				sprintf(dest, "%%%02X", (unsigned char) *src), dest += 3;
				nLenDest+=3;
			}
			*dest = 0;
		}
}

void urlDecode(const char *src, int nLenSrc, char* dest, int& nLenDest)
{
	unsigned int c = 0;
	const char* s = src;
	char* d = dest;
	int nMaxLenth = (0 == nLenDest?1024:nLenDest);
	nLenDest = 0;
	
	while ( s - src < nLenSrc  && nMaxLenth-- >= 0 )
	{
		if (*s != '%')
		{
			//if (*s != '+')
				*d++ = *s++;
			//else
			//{
			//	*d++ = ' ';
			//	s++;
			//}
		}
		else
		{
			sscanf(s, "%%%2X", &c);
			s += 3;
			*d++ = (unsigned char) c;
		}
		nLenDest++;
	}
}

std::string getUrlDecode(std::string strUrl)
{
	if (strUrl.empty())
	{
		return "";
	}
	int len = strUrl.length()+10;
	char *pUrlDecode = new char[len];
	memset(pUrlDecode,0,len);
	urlDecode(strUrl.c_str(),strUrl.length(),pUrlDecode,len);
	string strUrlDecode = pUrlDecode;
	delete[] pUrlDecode;
	return strUrlDecode;
}

std::string getUrlEncode(std::string strUrl)
{
	if (strUrl.empty())
	{
		return "";
	}
	int len = strUrl.length()*3+10;
	char *pUrlEncode = new char[len];
	memset(pUrlEncode,0,len);
	urlEncode(strUrl.c_str(),strUrl.length(),pUrlEncode,len);
	string strUrlEncode = pUrlEncode;
	delete[] pUrlEncode;
	return strUrlEncode;
}

std::string getBase64Decode(std::string strUrl)
{
	int iEncLen = strUrl.length();
	int iEncDataLen = Base64::GetDataLength(iEncLen);
	char *pDecBase64 = new char[iEncDataLen+1];	
	int iRet = Base64::Decode((char *)strUrl.c_str(),iEncLen,pDecBase64);
	pDecBase64[iRet] = '\0';
	string strRet = pDecBase64;
	delete[] pDecBase64;
	return strRet;
}

std::string getBase64Encode(std::string strUrl)
{
	int iEncLen = strUrl.length();
	int iEncCodeLen = Base64::GetCodeLength(iEncLen);
	char *pEncBase64 = new char[iEncCodeLen+1];	
	int iRet = Base64::Encode((char *)strUrl.c_str(),iEncLen,pEncBase64);
	pEncBase64[iRet] = '\0';
	string strRet = pEncBase64;
	delete[] pEncBase64;
	return strRet;
}

std::string hash2Char(const unsigned char* hash)
{
	char buf[41]={0};
	int i,j=0;
	unsigned char c;
	for(i=0;i<20;)
	{
		c = (hash[i]>>4) & 0x0f;
		if(c > 9)
			buf[j++] = 'A'+ (c - 10);
		else
			buf[j++] = '0' + c;
		c = hash[i] & 0x0f;
		if(c > 9)
			buf[j++] = 'A'+ (c - 10);
		else
			buf[j++] = '0' + c;
		i++;
	}
	return buf;
}

void char2Hash(const char* chars,unsigned char* hash)
{
	int i ,j=0;
	unsigned char c;
	
	for (i=0;i<40;)
	{
		if('9'>= chars[i]&& chars[i]>= '0' ) 
			c = (chars[i] - '0')<< 4;
		else if('F'>= chars[i]&& chars[i]>= 'A' )
			c = (chars[i] - 'A'+10)<< 4;
		else if('f'>= chars[i]&& chars[i]>= 'a')
			c = (chars[i] - 'a'+10)<< 4;
		else
			return ;
		i++;
		if('9'>= chars[i]&& chars[i]>= '0' ) 
			c += (chars[i] - '0');
		else if('F'>= chars[i]&& chars[i]>= 'A' )
			c += (chars[i] - 'A'+10);
		else if('f'>= chars[i]&& chars[i]>= 'a')
			c += (chars[i] - 'a'+10);
		else
			return ;
		i++;
		hash[j]= c;
		j++;
	}
}

void getStrHash(char* input,int len,char* hash)
{
	CSHA1 sha;
	sha.reset();
	sha.write(input,len);
	sha.read(hash);
}

//真HASH转化为假HASH
void trueHash2FalseHash(unsigned char hash[20])
{
	unsigned int* phash = (unsigned int*)hash;
	int m,n = 0;
	const char key[4] = {1,9,7,8};
	int range = hash[2]%4+1;

	for(m = 0;m<5;m++)
	{
		phash[m]  = (phash[m]>>key[n]) + (phash[m]<<(32-key[n]));
		hash[0+m*4] ^= 0x69;
		hash[1+m*4] ^= 0x4A;
		hash[2+m*4] ^= 0x87;
		hash[3+m*4] ^= 0x3C;
		n = (n+1)%range;
	}
}

//假HASH转化为真HASH
void falseHash2TrueHash(unsigned char hash[20])
{
	unsigned int *phash = (unsigned int*)hash;
	int m = 0;
	int n = 0;
	const char key[4] = {1,9,7,8};
	for(m = 0;m<5;m++)
	{
		hash[0+m*4] ^= 0x69;
		hash[1+m*4] ^= 0x4A;
		hash[2+m*4] ^= 0x87;
		hash[3+m*4] ^= 0x3C;
		phash[m] = (phash[m]<<key[n]) + (phash[m]>>(32-key[n]));
		int range = hash[2]%4+1;
		n = (n+1)%range;
	}
}

unsigned long long ntohll(unsigned long long val)
{
	if (__BYTE_ORDER == __LITTLE_ENDIAN)
	{
		return (((unsigned long long )htonl((int)((val << 32) >> 32))) << 32) | (unsigned int)htonl((int)(val >> 32));
	}
	else if (__BYTE_ORDER == __BIG_ENDIAN)
	{
		return val;
	}
}

unsigned long long htonll(unsigned long long val)
{
	if (__BYTE_ORDER == __LITTLE_ENDIAN)
	{
		return (((unsigned long long )htonl((int)((val << 32) >> 32))) << 32) | (unsigned int)htonl((int)(val >> 32));
	}
	else if (__BYTE_ORDER == __BIG_ENDIAN)
	{
		return val;
	}
}

unsigned long getTickCount(){
	int res;
	struct timespec sNow;
	res = clock_gettime(CLOCK_MONOTONIC, &sNow);
	if(res != 0)
	{
		printf("clock_gettime error: %d", errno);
		return -1;
	}
	return sNow.tv_sec * 1000 + sNow.tv_nsec / 1000000; /* milliseconds */
}

/* return time string */
int getTimeStr(char *dstBuf)
{
#ifdef WIN32 /* WIN32 */
	SYSTEMTIME st;
	GetLocalTime(&st);
	sprintf(dstBuf," %04d-%02d-%02d %02d:%02d:%02d.%03d ",
		st.wYear, st.wMonth, st.wDay, st.wHour,
		st.wMinute, st.wSecond, st.wMilliseconds);
	return st.wDay;
#else /* posix */
	time_t t;
	struct tm st;
	t = time(NULL);
	localtime_r(&t, &st);
	sprintf(dstBuf," %04d-%02d-%02d %02d:%02d:%02d.%03lu ",
		st.tm_year + 1900, st.tm_mon + 1, st.tm_wday, st.tm_hour,
		st.tm_min, st.tm_sec, getTickCount()%1000);
	return st.tm_wday;
#endif /* posix end */
}

long long getTimeUnix()
{
#ifdef WIN32 /* WIN32 */
	/*
	 * the number of milliseconds that
	 * have elapsed since the system was started.
	 */
	return GetTickCount(); /* milliseconds */
#else /* posix */
	/*int res;
	struct timespec sNow;
	res = clock_gettime(CLOCK_MONOTONIC, &sNow);
	if(res != 0)
	{
		// error 
		return 0;
	}
	long long uptime = sNow.tv_sec;
	uptime = uptime*1000;
	uptime += sNow.tv_nsec/1000000;
	return uptime; */
	long long tt = (long long)time(NULL);
	return tt;
#endif /* posix end */
}

int getTimeDay()
{
	time_t t;
	struct tm st;
	t = time(NULL);
	localtime_r(&t, &st);
	return st.tm_wday;
}

void getDateTime(char* szDTime)
{
	time_t t;
	struct tm st;
	t = time(NULL);
	localtime_r(&t, &st);
	sprintf(szDTime,"%04d-%02d-%02d",
		st.tm_year+1900,st.tm_mon+1, st.tm_mday);
}

int cmsMkdir(const char *dirname)
{
	int res;
	char path[256] = {0};
	memcpy(path, dirname, strlen(dirname));
#ifdef WIN32 /* WIN32 */
	char* pEnd = strstr(path, "\\");
	if(NULL == pEnd)
	{
		res = _mkdir(path);
	}
	while(pEnd)
	{
		*pEnd = 0;
		res = _mkdir(path);
		*pEnd = '\\';
		if(NULL == strstr(pEnd + 1, "\\"))
		{
			if('\0' != *(pEnd+1))
			{
				_mkdir(path);
			}
			break;
		}
		else
		{
			pEnd = strstr(pEnd + 1, "\\");
		}
	}
#else /* posix */
	char* pEnd = strstr(path+1, "/");
	if(NULL == pEnd)
	{
		res = mkdir(path, 0777);
	}
	while(pEnd)
	{
		*pEnd = 0;
		res = mkdir(path, 0777);
		*pEnd = '/';
		if(NULL == strstr(pEnd + 1, "/"))
		{
			if('\0' != *(pEnd+1))
			{
				mkdir(path, 0777);
			}
			break;
		}
		else
		{
			pEnd = strstr(pEnd + 1, "/");
		}
	}
#endif /* posix end */
	if(-1 == res) {
		/* error */
		return -1;
	}
	return 0;
}

bool isLegalIp(const char * const szIp)
{
	// 	return true;
	const char *pPos = szIp;
	int iIPnum = 0;
	int iCount = 0;
	while ( iCount < 4 )
	{
		++iCount;
		iIPnum = 0;
		int bit = 0;
		while(*pPos >= '0' && *pPos <= '9' )
		{
			iIPnum = iIPnum * 10 + (*pPos - '0');
			++pPos;
			++bit;
		}
		if (bit == 0)
		{
			return false;
		}
		if ( !(iIPnum >= 0 && iIPnum <= 255) )
		{
			return false;
		}
		if ( iCount < 4 && *pPos != '.')
		{
			return false;
		}
		++pPos;
	}
	return true;
}

unsigned long ipStr2ipInt(const char* szIP)
{
	return inet_addr(szIP);
}

void ipInt2ipStr(unsigned long iIp,char* szIP)
{
	unsigned char ip[4];
	memcpy(ip,(void*)&iIp,4);
	size_t pos=0;
	for(int i=0;i<4;i++)
	{
		pos+=sprintf(szIP+pos, "%d", ip[i]);
		if(i!=3)
		{
			szIP[pos]='.';
			pos+=1;		
		}
	}
}

std::string readMajorUrl(std::string strUrl) 
{
	size_t pos = strUrl.find("?");
	if (pos != std::string::npos)
	{
		strUrl = strUrl.substr(0,pos);
	}
	LinkUrl linkUrl;
	if (parseUrl(strUrl,linkUrl))
	{
		if (isLegalIp(linkUrl.host.c_str()))
		{
			if (linkUrl.protocol == PROTOCOL_HTTP)
			{
				strUrl = "http://";
			}
			if (linkUrl.protocol == PROTOCOL_HTTPS)
			{
				strUrl = "https://";
			}
			else if (linkUrl.protocol == PROTOCOL_RTMP)
			{
				strUrl = "rtmp://";
			}
			strUrl += linkUrl.app;
			strUrl += "/";
			strUrl += linkUrl.instanceName;
		}
	}
	return strUrl;
}

std::string readHashUrl(std::string strUrl)
{
	size_t stPos = strUrl.find("?");
	if (stPos != string::npos)
	{
		strUrl = strUrl.substr(0,stPos);
	}
	stPos = strUrl.find("://");
	if (stPos == string::npos)
	{
		return "";
	}
	else if (stPos == 0)
	{
		return "";
	}	
	string  strProtocol = strUrl.substr(0,stPos);
	transform(strProtocol.begin(), strProtocol.end(), strProtocol.begin(), ::tolower);
	size_t stBegin = stPos + 3;
	stPos = strUrl.find("/",stBegin);
	if (stPos == string::npos)
	{
		return "";
	}
	string strPath = strUrl.substr(stPos);
	string strHost = strUrl.substr(stBegin,stPos-stBegin);
	stBegin = stPos+1;
	stPos = strHost.find(":");
	if (stPos != string::npos)
	{
		strHost = strHost.substr(0,stPos);
	}
	if (!isLegalIp(strHost.c_str()))
	{
		strHost = readMainHost(strHost);
	}
	stPos = strPath.find(".flv");
	if (stPos != string::npos)
	{
		strPath = strPath.substr(0,stPos);
	}
	strUrl = strHost;
	strUrl += strPath;
	return strUrl;
}

std::string readMainHost(std::string strHost)
{
	string major;
	string com = strHost;
	size_t pos = com.find(".");
	if (pos == string::npos)
	{
		logs->error("*** [readMainHost] host %s error ***\n",strHost.c_str());
		return "";
	}
	while (pos != string::npos)
	{
		major = com.substr(0,pos);
		com = com.substr(pos+1);
		pos = com.find(".");
	}
	major.append(".");
	major.append(com);
	return major;
}

void split(std::string& s, std::string& delim,std::vector< std::string > &ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim,last);
	while (index != std::string::npos)
	{
		ret.push_back(s.substr(last,index-last));
		last = index+1;
		index = s.find_first_of(delim,last);
	}
	if (index-last > 0)
	{
		ret.push_back(s.substr(last,index-last));
	}  
}

std::string trim(std::string &s,std::string &delim)
{  
	if (s.empty())
	{
		return s;
	}
	s.erase(0,s.find_first_not_of(delim));
	s.erase(s.find_last_not_of(delim) + 1);
	return s;
}

HASH makeHash(const char *bytes,int len)
{
	CSHA1 sha;
	sha.write(bytes, len);
	string strHash = sha.read();
	HASH hash = HASH((char *)strHash.c_str());
	return hash;
}

const char *g_Speed[] = {(char *)"Byte/s", "KB/s", "MB/s", "GB/s"};
const char *g_Mem[] = {"Byte", "KB", "MB", "GB"};
std::string parseSpeed8Mem(int64 speed,bool isSpeed)
{
	float fSpeed = (float)speed;
	int i = 0;
	for (; i < 4;)
	{
		float sp = fSpeed / (float)(1024);
		if (int64(sp) == 0)
		{
			break;
		}
		fSpeed = sp;
		i++;
	}
	char szValue[128] = {0};
	snprintf(szValue,sizeof(szValue),"%.02f",fSpeed);
	std::string value = szValue;
	if (isSpeed)
	{
		value += g_Speed[i];
	}
	else
	{
		value += g_Mem[i];
	}
	return value;
}