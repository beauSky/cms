#ifndef __CMS_UTILITY_H__
#define __CMS_UTILITY_H__
#include <string>
#include <vector>
#include <common/cms_type.h>

#define CMS_OK 0
#define CMS_ERROR -1

#ifdef WIN32 /* WIN32 */
#define CMS_INVALID_SOCK INVALID_SOCKET 
#else /* posix */
#define CMS_INVALID_SOCK -1
#endif /* posix end */

#ifndef cmsMin
#define cmsMin(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifdef WIN32 /* WIN32 */
#include <windows.h>
#define cmsSleep(e) Sleep(e)
#else /* posix */
#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>
#define gettid() syscall(__NR_gettid)
#define _atoi64(val) strtoll(val, NULL, 10)
#define hex2int64(val) strtoll(val, NULL, 16)
#define cmsSleep(e) usleep(e*1000)
#endif /* posix end */

void getStrHash(char* input,int len,char* hash);
void trueHash2FalseHash(unsigned char hash[20]);
void falseHash2TrueHash(unsigned char hash[20]);
std::string hash2Char(const unsigned char* hash);
void char2Hash(const char* chars,unsigned char* hash);
void urlEncode(const char *src, int nLenSrc, char *dest, int& nLenDest);
void urlDecode(const char *src, int nLenSrc, char* dest, int& nLenDest);
std::string getUrlDecode(std::string strUrl);
std::string getUrlEncode(std::string strUrl);
std::string getBase64Decode(std::string strUrl);
std::string getBase64Encode(std::string strUrl);
unsigned long long ntohll(unsigned long long val);
unsigned long long htonll(unsigned long long val);
unsigned long getTickCount();
int getTimeStr(char *dstBuf);
void getDateTime(char* szDTime);
long long getTimeUnix();
int getTimeDay();
int cmsMkdir(const char *dirname);
bool isLegalIp(const char * const szIp);
unsigned long ipStr2ipInt(const char* szIP);
void ipInt2ipStr(unsigned long iIp,char* szIP);
std::string readMajorUrl(std::string strUrl);
std::string readHashUrl(std::string strUrl);
std::string readMainHost(std::string strHost);
void split(std::string& s, std::string& delim,std::vector< std::string > &ret);
std::string trim(std::string &s,std::string &delim);
HASH makeHash(const char *bytes,int len);
std::string parseSpeed8Mem(int64 speed,bool isSpeed);
#endif