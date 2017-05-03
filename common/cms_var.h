#ifndef __CMS_COMMON_TYPE_H__
#define __CMS_COMMON_TYPE_H__
#include <libev/ev.h>

typedef void (*cms_timer_cb)(void *t);
typedef struct _cms_timer 
{ 
	int fd;
	long long tick;
	int	only;				//0 ��ʾû��ʹ�ã�����0��ʾ���ڱ�ʹ�ô���
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