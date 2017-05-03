#ifndef __CMS_COMMON_VAR_H__
#define __CMS_COMMON_VAR_H__
#include <string.h>

typedef long long			int64;
typedef unsigned long long	uint64;
typedef int					int32;
typedef unsigned int		uint32;
typedef short				int16;
typedef unsigned short		uint16;
typedef char				int8;
typedef unsigned char		uint8;
typedef unsigned char		byte;

typedef struct _HASH{
public:
	unsigned char data[20];
	_HASH()
	{
		memset(data,0,20);
	}
	_HASH(char* hash)
	{
		memcpy(data,hash,20);
	}
	_HASH(unsigned char* hash)
	{
		memcpy(data,hash,20);
	}
	bool operator < (_HASH const& _A) const
	{
		if(memcmp(data, _A.data, 20) < 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	bool operator == (_HASH const& _A) const
	{
		if(memcmp(data,_A.data,20)==0)
			return true;
		else
			return false;
	}
	bool operator != (_HASH const& _A) const
	{
		if(memcmp(data,_A.data,20)==0)
			return false;
		else
			return true;
	}
	_HASH& operator = (_HASH const& _A) 
	{
		memcpy(data,_A.data,20);
		return *this;
	}
} HASH;

#endif