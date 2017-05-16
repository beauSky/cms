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
#ifndef __CMS_INTERFACE_CONN_H__
#define __CMS_INTERFACE_CONN_H__
#include <string>
#include <common/cms_var.h>

class Conn
{
public:
	Conn();
	virtual ~Conn();
	virtual int doit() = 0;
	virtual int handleEv(FdEvents *fe) = 0;
	virtual int stop(std::string reason) = 0;
	virtual std::string getUrl() = 0;
	virtual std::string getPushUrl() = 0;
	virtual std::string getRemoteIP() = 0;

	virtual cms_net_ev    *evReadIO() = 0;
	virtual cms_net_ev    *evWriteIO() = 0;

	virtual void down8upBytes() = 0;

	//http สนำร
	virtual int doDecode() = 0;
	virtual int doReadData() = 0; //http client
	virtual int doTransmission() = 0; //http server
	virtual int sendBefore(const char *data,int len) = 0;
};

#endif
