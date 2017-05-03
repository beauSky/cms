#ifndef __CMS_THREAD_H__
#define __CMS_THREAD_H__

class CThread 
{
public:
	CThread();
	virtual ~CThread();
	virtual int  start() = 0;
	virtual void run() = 0;
	virtual int  end() = 0;
};

void createThread(CThread *pthread);
#endif
