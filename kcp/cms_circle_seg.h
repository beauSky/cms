#ifndef __CMS_CIRCLE_H__
#define __CMS_CIRCLE_H__
#include <kcp/ikcp.h>

typedef struct IKCPSEG IkcpSeg;
typedef struct _CmsCircleSeg 
{
	IkcpSeg			**mptr;
	unsigned int	mb;			//起始位置
	unsigned int	me;			//结束位置
	unsigned int	mn;			//总数
	unsigned int	mu;			//使用数   mu == 0 或者 mb == me 时 为空
}CmsCircleSeg;


//ccsCreate 创建循环内存数组 size指定大小
CmsCircleSeg *ccsCreate(unsigned int size);
//ccsRelease 释放循环内存数组
void ccsRelease(CmsCircleSeg *ccs);


#endif
