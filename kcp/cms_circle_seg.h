#ifndef __CMS_CIRCLE_H__
#define __CMS_CIRCLE_H__
#include <kcp/ikcp.h>

typedef struct IKCPSEG IkcpSeg;
typedef struct _CmsCircleSeg 
{
	IkcpSeg			**mptr;
	unsigned int	mb;			//��ʼλ��
	unsigned int	me;			//����λ��
	unsigned int	mn;			//����
	unsigned int	mu;			//ʹ����   mu == 0 ���� mb == me ʱ Ϊ��
}CmsCircleSeg;


//ccsCreate ����ѭ���ڴ����� sizeָ����С
CmsCircleSeg *ccsCreate(unsigned int size);
//ccsRelease �ͷ�ѭ���ڴ�����
void ccsRelease(CmsCircleSeg *ccs);


#endif
