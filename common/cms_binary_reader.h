
#ifndef BinaryReader_INCLUDED
#define BinaryReader_INCLUDED

#include <istream>

class  BinaryReader
{
public:

	BinaryReader(int sock);

	~BinaryReader();

	BinaryReader& operator >> (bool& value);
	BinaryReader& operator >> (char& value);
	BinaryReader& operator >> (unsigned char& value);
	BinaryReader& operator >> (signed char& value);
	BinaryReader& operator >> (short& value);
	BinaryReader& operator >> (unsigned short& value);
	BinaryReader& operator >> (int& value);
	BinaryReader& operator >> (unsigned int& value);
	BinaryReader& operator >> (long& value);
	BinaryReader& operator >> (unsigned long& value);
	BinaryReader& operator >> (float& value);
	BinaryReader& operator >> (double& value);
	BinaryReader& operator >> (long long& value);
	

	BinaryReader& operator >> (std::string& value);

	void read7BitEncoded(int& value);

	void readRaw(char* value,int length);

private:
	char* m_pHead;
	char* m_pPos;
	int   m_iLen;
	bool  m_bError;
};




#endif 
