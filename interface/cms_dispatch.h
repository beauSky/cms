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
#ifndef __CMS_DISPATCH_H__
#define __CMS_DISPATCH_H__
#include <net/cms_net_var.h>
#include <ev/cms_ev.h>

class CDispatch 
{
public:
	CDispatch();
	virtual ~CDispatch();
	virtual void  pushEv(int,int,cms_net_ev *watcherRead,cms_net_ev *watcherWrite) = 0;
	virtual void  pushEv(int,int,cms_timer *ct) = 0;
};
#endif
