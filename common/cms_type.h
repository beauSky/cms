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
#ifndef __CMS_COMMON_VAR_H__
#define __CMS_COMMON_VAR_H__
#include <string.h>

enum ConnType
{
	TypeNetNone,
	TypeHttp,
	TypeHttps,
	TypeRtmp,
	TypeQuery
};

#ifndef CMS_BASIC_TYPE
#define CMS_BASIC_TYPE
typedef long long			int64;
typedef unsigned long long	uint64;
typedef int					int32;
typedef unsigned int		uint32;
typedef short				int16;
typedef unsigned short		uint16;
typedef char				int8;
typedef unsigned char		uint8;
typedef unsigned char		byte;
#endif

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