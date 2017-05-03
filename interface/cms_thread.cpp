#include <interface/cms_thread.h>
#include <common/cms_utility.h>

#define ThreadJoinable 1
#define ThreadUnJoinable 0

void* createThread(void *arg)
{
	CThread *pthread = (CThread*)arg;
	if (pthread->start() == CMS_ERROR)
	{
		return NULL;
	}
	pthread->run();
	pthread->end();
	return NULL;
}

CThread::CThread()
{

}

CThread::~CThread()
{

}

