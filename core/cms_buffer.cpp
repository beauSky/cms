#include <core/cms_buffer.h>
#include <common/cms_utility.h>
#include <core/cms_errno.h>
#include <log/cms_log.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

CBufferReader::CBufferReader(CReaderWriter *rd,int size)
{
	mbufferSize = size;
	mbuffer = (char *)malloc(size);
	mb = me = 0;
	mrd = rd;
	ms2nConn = NULL;
	mtotalReadBytes = 0;
}

CBufferReader::CBufferReader(s2n_connection *s2nConn,int size)
{
	mbufferSize = size;
	mbuffer = (char *)malloc(size);
	mb = me = 0;
	mrd = NULL;
	ms2nConn = s2nConn;
	mtotalReadBytes = 0;
}

CBufferReader::~CBufferReader()
{
	free(mbuffer);
	mbuffer = NULL;
	mb = me = 0;
	mbufferSize = 0;
}

int   CBufferReader::grow(int n)
{
	int outread = 0;
	int nuseBuffer = me-mb;
	/*if (nuseBuffer >= n)
	{
		return CMS_OK;
	}*/
	int ret = 0;
	int nfreeBuffer = mbufferSize-nuseBuffer;
	int ncanUserBuffer = mbufferSize-me;
	int needRead = n/* - nuseBuffer*/;//需要读数据的长度
	int nread = 0;
	if (mbufferSize < n || nfreeBuffer < n)//需要扩展内存
	{
		mbufferSize += ((n / 1024 + 1) * 1024);
		resize();		
	}
	else
	{
		if (nuseBuffer == 0)
		{
			//内存没有使用，重设
			mb = me = 0;
		}
		else if (ncanUserBuffer < needRead)
		{
			memmove(mbuffer,mbuffer+mb,me-mb);
			me = me-mb;
			mb = 0;			
			ncanUserBuffer = mbufferSize-me;
		}
	}
	while (needRead > 0)
	{
		if (ms2nConn == NULL)
		{
			ret = mrd->read(mbuffer+me,needRead,nread);
			if (ret != CMS_OK)
			{
				logs->error("*** buffer reader read fail,errno=%d,errstr=%s ***",mrd->errnos(),mrd->errnoCode());
				merrcode = mrd->errnos();
				return CMS_ERROR;
			}
		}
		else if (mrd == NULL)
		{
			s2n_blocked_status blocked;
			nread = s2n_recv(ms2nConn, mbuffer+me, needRead, &blocked);
			if (nread == 0) {
				/* Connection has been closed */
				merrcode = CMS_ERRNO_FIN;
				logs->error("*** tls buffer reader s2n_recv fail,errno=%d,errstr=%s ***",merrcode,cmsStrErrno(merrcode));
				return CMS_ERROR;
			}
			if (nread < 0) 
			{
				logs->error("*** tls buffer reader s2n_recv error from connection: '%s' %d\n", s2n_strerror(s2n_errno, "EN"), s2n_connection_get_alert(ms2nConn));
				seterrno(s2n_errno);
				return CMS_ERROR;
			}
		}		
		if (nread == 0)
		{
			break;
		}
		mtotalReadBytes += nread;
		needRead -= nread;
		me += nread;
		outread += nread;
		nread = 0;		
		break;
	}
	//尝试多读数据
// 	if (mbufferSize-me > 0)
// 	{
// 		nread = 0;
// 		ret = mrd->read(mbuffer+me,mbufferSize-me,nread);
// 		if (ret == CMS_OK && nread > 0)
// 		{
// 			me += nread;
// 			outread += nread;
// 		}
// 	}
	return CMS_OK;
}

char  *CBufferReader::readBytes(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	char *ptr = mbuffer+mb;
	mb += n;
	if (mb==me)
	{
		mb = me = 0;
	}
	return ptr;
}

char  *CBufferReader::peek(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	return mbuffer+mb;
}

char   CBufferReader::readByte()
{
	assert(mbuffer);
	assert(me-mb>=1);
	char ch = mbuffer[mb];
	mb++;
	if (mb==me)
	{
		mb = me = 0;
	}
	return ch;
}

void  CBufferReader::skip(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	mb += n;
}

int   CBufferReader::size()
{
	return  me-mb;
}

void   CBufferReader::resize()
{
	assert(mbufferSize>=0);
	mbuffer = (char *)realloc(mbuffer,mbufferSize);
}

char  *CBufferReader::errnoCode()
{
	return cmsStrErrno(merrcode);
}

int  CBufferReader::errnos()
{
	return merrcode;
}

int CBufferReader::seterrno(int err)
{
	switch (err)
	{
	case S2N_ERR_T_IO:
		merrcode = CMS_S2N_ERR_T_IO;
		break;
	case S2N_ERR_T_CLOSED:
		merrcode = CMS_S2N_ERR_T_CLOSED;
		break;
	case S2N_ERR_T_BLOCKED:
		merrcode = CMS_S2N_ERR_T_BLOCKED;
		break;
	case S2N_ERR_T_ALERT:
		merrcode = CMS_S2N_ERR_T_ALERT;
		break;
	case S2N_ERR_T_PROTO:
		merrcode = CMS_S2N_ERR_T_PROTO;
		break;
	case S2N_ERR_T_INTERNAL:
		merrcode = CMS_S2N_ERR_T_INTERNAL;
		break;
	case S2N_ERR_T_USAGE:
		merrcode = CMS_S2N_ERR_T_USAGE;
		break;
	default:
		merrcode = CMS_ERROR_UNKNOW;
		break;
	}
	return 0;
}

void  CBufferReader::close()
{
	mrd->close();
}

int32 CBufferReader::readBytesNum()
{
	int32 bytes = mtotalReadBytes;
	mtotalReadBytes = 0;
	return bytes;
}

CBufferWriter::CBufferWriter(CReaderWriter *rd,int size)
{
	mbufferSize = size;
	mbuffer = (char *)malloc(size);
	mb = me = 0;
	mrd = rd;
	ms2nConn = NULL;
	mtotalWriteBytes = 0;
}

CBufferWriter::CBufferWriter(s2n_connection *s2nConn,int size)
{
	mbufferSize = size;
	mbuffer = (char *)malloc(size);
	mb = me = 0;
	mrd = NULL;
	ms2nConn = s2nConn;
	mtotalWriteBytes = 0;
}

CBufferWriter::~CBufferWriter()
{
	free(mbuffer);
	mbuffer = NULL;
	mb = me = 0;
	mbufferSize = 0;
}

int CBufferWriter::writeBytes(const char *data,int n)
{
	mtotalWriteBytes += n;
	int nn = n;
	int ret = 0;
	int nwrite = 0;
	const char *p = data;
	int nuseBuffer = me-mb;
	if (nuseBuffer > 0)
	{
		//先把以前的数据发了
		nwrite = 0;
		if (ms2nConn == NULL)
		{
			ret = mrd->write(mbuffer+mb,nuseBuffer,nwrite);
			if (ret != CMS_OK)
			{
				logs->error("*** buffer writer write fail,errno=%d,errstr=%s ***",mrd->errnos(),mrd->errnoCode());
				merrcode = mrd->errnos();
				return CMS_ERROR;
			}
		}
		else if (mrd == NULL)
		{
			s2n_blocked_status blocked;
			nwrite = s2n_send(ms2nConn, mbuffer+me, nuseBuffer, &blocked);
			if (nwrite == 0) 
			{
				/* Connection has been closed */
				merrcode = CMS_ERRNO_FIN;
				logs->error("*** tls buffer writer s2n_send fail,errno=%d,errstr=%s ***",merrcode,cmsStrErrno(merrcode));
				return CMS_ERROR;
			}
			if (nwrite < 0) 
			{
				logs->error("*** tls buffer writer s2n_send error from connection: '%s' %d\n", s2n_strerror(s2n_errno, "EN"), s2n_connection_get_alert(ms2nConn));
				seterrno(s2n_errno);
				return CMS_ERROR;
			}
		}
		mb += nwrite;
		nuseBuffer = me-mb;
	}
	if (nuseBuffer == 0)
	{
		mb = me = 0;
		//如果以前的数据发送完毕，那就发传进来的吧
		nwrite = 0;
		if (ms2nConn == NULL)
		{
			ret = mrd->write((char *)p,n,nwrite);
			if (ret != CMS_OK)
			{
				logs->error("*** buffer writer write fail,errno=%d,errstr=%s ***",mrd->errnos(),mrd->errnoCode());
				merrcode = mrd->errnos();
				return CMS_ERROR;
			}
		}
		else if (mrd == NULL)
		{
			s2n_blocked_status blocked;
			nwrite = s2n_send(ms2nConn, (char *)p,n, &blocked);
			if (nwrite == 0) 
			{
				/* Connection has been closed */
				merrcode = CMS_ERRNO_FIN;
				logs->error("*** tls buffer writer s2n_send fail,errno=%d,errstr=%s ***",merrcode,cmsStrErrno(merrcode));
				return CMS_ERROR;
			}
			if (nwrite < 0) 
			{
				logs->error("*** tls buffer writer s2n_send error from connection: '%s' %d\n", s2n_strerror(s2n_errno, "EN"), s2n_connection_get_alert(ms2nConn));
				seterrno(s2n_errno);
				return CMS_ERROR;
			}
		}
		p += nwrite;
		n -= nwrite;
	}

	int nfreeBuffer = mbufferSize-nuseBuffer;
	if (mbufferSize < n || nfreeBuffer < n)//需要扩展内存
	{
		mbufferSize += ((n / 1024 + 1) * 1024);
		resize();
	}
	int ncanUserBuffer = mbufferSize-me;	
	if (ncanUserBuffer < n)
	{
		memmove(mbuffer,mbuffer+mb,me-mb);
		mb = 0;
		me = nuseBuffer;
		ncanUserBuffer = mbufferSize-me;
	}
	assert(ncanUserBuffer>=n);
	if (n > 0)
	{
		memcpy(mbuffer+me,p,n);
		me += n;
	}
	return nn - n;
}

int CBufferWriter::writeByte(char ch)
{
	return writeBytes(&ch,1);
}

int CBufferWriter::flush()
{
	if (me-mb == 0)
	{
		return CMS_OK;
	}
	int nwrite = 0;
	if (ms2nConn == NULL)
	{
		int ret = mrd->write(mbuffer+mb,me-mb,nwrite);
		if (ret != CMS_OK)
		{
			logs->error("*** buffer writer flush fail,errno=%d,errstr=%s ***",mrd->errnos(),mrd->errnoCode());
			merrcode = mrd->errnos();
			return CMS_ERROR;
		}
	}
	else if (mrd == NULL)
	{
		s2n_blocked_status blocked;
		nwrite = s2n_send(ms2nConn, mbuffer+mb,me-mb, &blocked);
		if (nwrite == 0) 
		{
			/* Connection has been closed */
			merrcode = CMS_ERRNO_FIN;
			logs->error("*** tls buffer writer s2n_send fail,errno=%d,errstr=%s ***",merrcode,cmsStrErrno(merrcode));
			return CMS_ERROR;
		}
		if (nwrite < 0) 
		{
			logs->error("*** tls buffer writer s2n_send error from connection: '%s' %d", s2n_strerror(s2n_errno, "EN"), s2n_connection_get_alert(ms2nConn));
			seterrno(s2n_errno);
			return CMS_ERROR;
		}
	}
	mb += nwrite;
	return nwrite;
}

bool CBufferWriter::isUsable()
{
	if (mbufferSize <= 128*1024)
	{
		return true;
	}
	int nuseBuffer = me-mb;
	int nfreeBuffer = mbufferSize-nuseBuffer;
	if (mbufferSize > 256*1024 && nfreeBuffer*100/mbufferSize > 30)
	{
		return true;
	}
	else if (mbufferSize <= 256*1024 && nfreeBuffer*100/mbufferSize > 10)
	{
		return true;
	}
	return false;
}

void  CBufferWriter::resize()
{
	assert(mbufferSize >= 0);
	mbuffer = (char *)realloc(mbuffer,mbufferSize);
}

char *CBufferWriter::errnoCode()
{
	return cmsStrErrno(merrcode);
}

int CBufferWriter::errnos()
{
	return merrcode;
}

int CBufferWriter::seterrno(int err)
{
	switch (err)
	{
	case S2N_ERR_T_IO:
		merrcode = CMS_S2N_ERR_T_IO;
		break;
	case S2N_ERR_T_CLOSED:
		merrcode = CMS_S2N_ERR_T_CLOSED;
		break;
	case S2N_ERR_T_BLOCKED:
		merrcode = CMS_S2N_ERR_T_BLOCKED;
		break;
	case S2N_ERR_T_ALERT:
		merrcode = CMS_S2N_ERR_T_ALERT;
		break;
	case S2N_ERR_T_PROTO:
		merrcode = CMS_S2N_ERR_T_PROTO;
		break;
	case S2N_ERR_T_INTERNAL:
		merrcode = CMS_S2N_ERR_T_INTERNAL;
		break;
	case S2N_ERR_T_USAGE:
		merrcode = CMS_S2N_ERR_T_USAGE;
		break;
	default:
		merrcode = CMS_ERROR_UNKNOW;
		break;
	}
	return 0;
}

int CBufferWriter::size()
{
	return me-mb;
}

void CBufferWriter::close()
{
	mrd->close();
}

int32 CBufferWriter::writeBytesNum()
{
	int32 bytes = mtotalWriteBytes;
	mtotalWriteBytes = 0;
	return bytes;
}

CByteReaderWriter::CByteReaderWriter(int size)
{
	mbufferSize = size;
	mbuffer = (char *)malloc(size);
	mb = me = 0;
}

CByteReaderWriter::~CByteReaderWriter()
{
	free(mbuffer);
	mbuffer = NULL;
	mb = me = 0;
	mbufferSize = 0;
}

int CByteReaderWriter::writeBytes(const char *data,int n)
{
	const char *p = data;
	int nuseBuffer = me-mb;	
	if (nuseBuffer == 0)
	{
		mb = me = 0;
	}
	int nfreeBuffer = mbufferSize-nuseBuffer;
	if (mbufferSize < n || nfreeBuffer < n)//需要扩展内存
	{
		mbufferSize += ((n / 1024 + 1) * 1024);
		resize();
	}
	int ncanUserBuffer = mbufferSize-me;	
	if (ncanUserBuffer < n)
	{
		memmove(mbuffer,mbuffer+mb,me-mb);
		mb = 0;
		me = nuseBuffer;
		ncanUserBuffer = mbufferSize-me;
	}
	assert(ncanUserBuffer>=n);
	if (n > 0)
	{
		memcpy(mbuffer+me,p,n);
		me += n;
	}
	return n;
}

int CByteReaderWriter::writeByte(char ch)
{
	return writeBytes(&ch,1);
}

char  *CByteReaderWriter::readBytes(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	char *ptr = mbuffer+mb;
	mb += n;
	if (mb==me)
	{
		mb = me = 0;
	}
	return ptr;
}

char  CByteReaderWriter::readByte()
{
	assert(mbuffer);
	assert(me-mb>=1);
	char ch = mbuffer[mb];
	mb++;
	if (mb==me)
	{
		mb = me = 0;
	}
	return ch;
}

char  *CByteReaderWriter::peek(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	return mbuffer+mb;
}

void  CByteReaderWriter::skip(int n)
{
	assert(mbuffer);
	assert(me-mb>=n);
	mb += n;
}

void CByteReaderWriter::resize()
{
	assert(mbufferSize >= 0);
	mbuffer = (char *)realloc(mbuffer,mbufferSize);
}

int CByteReaderWriter::size()
{
	return me-mb;
}


