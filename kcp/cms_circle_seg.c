#include <stdlib.h>
#include <kcp/cms_circle_seg.h>
#include <assert.h>

void *ccsMalloc(unsigned int size)
{
	return malloc(sizeof(IkcpSeg*)*size);
}

void ccsFree(void *ptr)
{
	free(mptr);
}

CmsCircleSeg *ccsCreate(unsigned int size)
{
	assert(size > 0);
	CmsCircleSeg *ccs = (CmsCircleSeg*)malloc(sizeof(CmsCircleSeg));
	ccs->mb = 0;
	ccs->me = 0;
	ccs->mn = size;
	ccs->mu = 0;
	ccs->mptr = (IkcpSeg**)ccsMalloc(size);
	for (int i = 0; i < ccs->mn; i++)
	{
		ccs->mptr[i] = NULL;
	}
	return ccs;
}

void ccsRelease(CmsCircleSeg *ccs)
{
	if (ccs->mu > 0)
	{
		for (int i = 0; i < ccs->mn; i++)
		{
			if (ccs->mptr[i] != NULL)
			{
				ikcp_segment_delete(NULL, ccs->mptr[i]);
				ccs->mptr[i] = NULL;
			}
		}
	}
	ccsFree(ccs->mptr);
	free(ccs);
}

