#include <core/cms_errno.h>
#include <string.h>

char *gstrErrno[CMS_ERRNO_NONE-CMS_ERRNO_TIMEOUT]={
	(char *)"Timeout",
	(char *)"Connection has been EOF",
	(char *)"Underlying I/O operation failed, check system errno",
	(char *)"Connection has been EOF",
	(char *)"Underlying I/O operation would block",
	(char *)"Incoming Alert",
	(char *)"Failure in some part of the TLS protocol. Ex: CBC verification failure",
	(char *)"Error internal to s2n. A precondition could have failed",
	(char *)"User input error. Ex: Providing an invalid cipher preference version"
};

char *cmsStrErrno(int code)
{
	if (code >= CMS_ERRNO_TIMEOUT && code < CMS_ERRNO_NONE)
	{
		return gstrErrno[code-CMS_ERRNO_TIMEOUT];
	}
	return strerror(code);
}