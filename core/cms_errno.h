#ifndef __CMS_ERRNO_H__
#define __CMS_ERRNO_H__

enum CmsErrnoCode
{
	CMS_ERRNO_TIMEOUT = 10086100,
	CMS_ERRNO_FIN,	
	CMS_S2N_ERR_T_IO, /* Underlying I/O operation failed, check system errno */
	CMS_S2N_ERR_T_CLOSED, /* EOF */
	CMS_S2N_ERR_T_BLOCKED, /* Underlying I/O operation would block */
	CMS_S2N_ERR_T_ALERT, /* Incoming Alert */
	CMS_S2N_ERR_T_PROTO, /* Failure in some part of the TLS protocol. Ex: CBC verification failure */
	CMS_S2N_ERR_T_INTERNAL, /* Error internal to s2n. A precondition could have failed. */
	CMS_S2N_ERR_T_USAGE, /* User input error. Ex: Providing an invalid cipher preference version */
	CMS_ERROR_UNKNOW,
	CMS_ERRNO_NONE,
};

char *cmsStrErrno(int code);
#endif
