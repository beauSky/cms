#ifndef __CMS_COMMON_TYPE_H__
#define __CMS_COMMON_TYPE_H__
#include <libev/ev.h>

typedef void (*cms_timer_cb)(void *t);
typedef struct _cms_timer 
{ 
	int fd;
	long long tick;
	int	only;				//0 表示没被使用，大于0表示正在被使用次数
	cms_timer_cb cb;
}cms_timer;

struct FdEvents 
{
	int fd;
	int events;
	struct ev_loop *loop;
	struct ev_io *watcherReadIO;
	struct ev_io *watcherWriteIO;
	struct ev_timer *watcherTimeout;
	struct _cms_timer *watcherCmsTimer;
};
#endif