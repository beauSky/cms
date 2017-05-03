#ifndef BinaryWriter_INCLUDED
#define BinaryWriter_INCLUDED

#include <ostream>
#include <stdlib.h>
#include <string.h>
class  BinaryWriter
{
public:
	
	BinaryWriter();

	~BinaryWriter();

	BinaryWriter& operator << (bool value);
	BinaryWriter& operator << (char value);
	BinaryWriter& operator << (unsigned char value);
	BinaryWriter& operator << (signed char value);
	BinaryWriter& operator << (short value);
	BinaryWriter& operator << (unsigned short value);
	BinaryWriter& operator << (int value);
	BinaryWriter& operator << (unsigned int value);
	BinaryWriter& operator << (long value);
	BinaryWriter& operator << (unsigned long value);
	BinaryWriter& operator << (float value);
	BinaryWriter& operator << (double value);
	BinaryWriter& operator << (long long value);


	BinaryWriter& operator << (const std::string& value);
	BinaryWriter& operator << (const char* value);

	void writeRaw(const char* value,int length);
	void rrealloc(const char* pBuf,int iSize);

	char* getData() 
	{ 
		return m_pHead;
	}

	void reset()
	{
		m_iLen = 0;
		m_pPos = m_pHead;
	}

	int getLength() 
	{
		return m_iLen;
	}
	
private:
	char* m_pHead;
	char* m_pPos;
	int   m_iBufSize;
	int   m_iLen;
};




#endif 
