#include <common/cms_binary_reader.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdlib.h>

BinaryReader::BinaryReader(int sock)
{
	m_bError = false;
	m_iLen = 0;
	int iCurrLen = 0;
	m_pHead = (char*)malloc(1024);

	while ( iCurrLen <  m_iLen || m_iLen == 0 )
	{
		int iRecv = ::recv(sock,m_pHead+iCurrLen,1024,0);
		if ( iRecv != -1)
		{
			if ( iCurrLen == 0 )
				m_iLen = ntohl(*(int*)m_pHead);
			//修正m_iLen过大的错误
			if( m_iLen > 1024 )
				break;
			iCurrLen += iRecv;
			if ( iCurrLen < m_iLen)
				m_pHead = (char*)realloc(m_pHead,iCurrLen+1024);
			else
				break;
		}	
		else
		{
			if ( errno == EAGAIN || errno == EWOULDBLOCK)
				continue;
			else
			{
				m_bError = true;
				break;
			}
		}
	}
	m_pPos = m_pHead;
}


BinaryReader::~BinaryReader()
{
	free(m_pHead);
}


BinaryReader& BinaryReader::operator >> (bool& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (char& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (unsigned char& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (signed char& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (short& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (unsigned short& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (int& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (unsigned int& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (long& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (unsigned long& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (float& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}


BinaryReader& BinaryReader::operator >> (double& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}

BinaryReader& BinaryReader::operator >> (long long& value)
{
	if ( !m_bError )
	{
		memcpy((char*) &value,m_pPos, sizeof(value));
		m_pPos += sizeof(value);
	}
	return *this;
}

BinaryReader& BinaryReader::operator >> (std::string& value)
{
	if ( !m_bError )
	{
		while(*m_pPos!='\0')
		{
			value += *m_pPos;
			m_pPos++;
		}
		m_pPos++;
	}
	return *this;
}



void BinaryReader::readRaw(char* value,int length)
{
	if ( !m_bError )
	{
		memcpy(value,m_pPos, length);
		m_pPos += length;
	}
}
