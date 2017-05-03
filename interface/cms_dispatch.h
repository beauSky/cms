#ifndef __CMS_DISPATCH_H__
#define __CMS_DISPATCH_H__
#include <libev/ev.h>
#include <ev/cms_ev.h>

class CDispatch 
{
public:
	CDispatch();
	virtual ~CDispatch();
	virtual void  pushEv(int,int,struct ev_loop *loop,struct ev_io *watcherRead,struct ev_io *watcherWrite,struct ev_timer *watcherTimeout) = 0;
	virtual void  pushEv(int,int,cms_timer *ct) = 0;
};
#endif
