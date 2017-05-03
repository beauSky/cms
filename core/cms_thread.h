#ifndef __CMS_THREAD_H__
#define __CMS_THREAD_H__
#ifdef WIN32 /* WIN32 */
#include <process.h>
#define CMS_THREAD_RETURN void
typedef	unsigned long cms_thread_t;
typedef void(__cdecl *cms_routine_pt)(void*);
#else /* posix */

#include <pthread.h>

#define CMS_THREAD_RETURN void*
typedef pthread_t cms_thread_t;
typedef void *(*cms_routine_pt)(void*);

#endif /* posix end */



/* func */
int cmsCreateThread(cms_thread_t *tid, cms_routine_pt routine, void *arg,bool detached);
int cmsWaitForThread(cms_thread_t tid, void **value_ptr);
int cmsWaitForMultiThreads(int nCount, const cms_thread_t *handles);


#endif /* _QVOD_THREAD_H_ */


