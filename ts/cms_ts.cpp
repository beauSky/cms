/*
The MIT License (MIT)

Copyright (c) 2017- cms(hsc)

Author: hsc/kisslovecsh@foxmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <ts/cms_ts.h>
#include <common/cms_utility.h>
#include <protocol/cms_amf0.h>
#include <log/cms_log.h>
#include <enc/cms_crc32.h>
#include <string>
#include <string.h>
#include <assert.h>

byte gSDT[] = {0x47, 0x40, 0x11, 0x10, 0x00, 0x42, 0xf0, 0x25, 0x00, 0x01, 0xc1, 0x00, 0x00, 0x00, 0x01, 0xff, 0x00, 0x01, 0xfc, 0x80,
	0x14, 0x48, 0x12, 0x01, 0x06, 0x46, 0x46, 0x6d, 0x70, 0x65, 0x67, 0x09, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x30,
	0x31, 0xa7, 0x79, 0xa0, 0x03, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

//����FLVͷ��Ϣ
SHead dealHead(byte *inBuf, int inLen)  
{
	SHead head = {0};
	head.mversion = inBuf[3];
	head.mstreamInfo = inBuf[4];
	head.mlenght = ((int)(inBuf[5]))<<24 | ((int)(inBuf[6]))<<16 | ((int)(inBuf[7]))<<8 | (int)(inBuf[8]);
	return head;
}

//����audioTag
SAudioInfo dealATag(byte *inBuf, int inLen)  
{
	SAudioInfo aInfo = {0};
	aInfo.mcodeType = inBuf[0] >> 4;
	aInfo.mrate = (inBuf[0] << 4) >> 6;
	aInfo.mprecision = (inBuf[0] << 6) >> 7;
	aInfo.maudioType = (inBuf[0] << 7) >> 7;
	return aInfo;
}

//����videoTag
SVideoInfo dealVTag(byte *inBuf, int inLen)  
{
	SVideoInfo vInfo = {0};
	vInfo.mframType = inBuf[0] >> 4;
	vInfo.mcodeId = (inBuf[0] << 4) >> 4;
	return vInfo;
}

//����TagSize
void dealTagSize(byte *inBuf, int inLen,int &tagSize,int &pTagSize) 
{
	pTagSize = ((int)(inBuf[0]))<<24 | ((int)(inBuf[1]))<<16 | ((int)(inBuf[2]))<<8 | (int)(inBuf[3]);
	tagSize = ((int)(inBuf[5]))<<16 | ((int)(inBuf[6]))<<8 | (int)(inBuf[7]);
	tagSize += 11;
}

//����scriptTag
SDataInfo dealSTag(byte *inBuf, int inLen)
{
	//������һ��AMF,һ���һ��AMFΪ�̶���Ϣ
	SDataInfo dInfo = {0};
	byte amfType = inBuf[13]; //
	if (amfType == 1)
	{
		//ʾ����������
	}
	int amfSize = int(inBuf[14])<<24 | int(inBuf[15])<<16 | int(inBuf[16])<<8 | int(inBuf[17]);

	float temp = 0;	
		
	amf0::Amf0Block *block = NULL;
	if (inLen-15 >= amfSize)
	{
		block = amf0::amf0Parse((char *)(inBuf+15), amfSize);
	}
	if (block != NULL)
	{
		string strValue;
		amf0::amf0Block5Value(block,"duration",strValue);
		if (!strValue.empty()) 
		{
			temp = str2float(strValue.c_str());
			dInfo.mduration = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"width",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mwidth = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"height",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mheight = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"videodatarate",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mvideodatarate = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"framerate",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mframerate = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"videocodecid",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mvideocodecid = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"audiosamplerate",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.maudiosamplerate = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"audiosamplesize",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.maudiosamplesize = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"stereo",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mstereo = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"audiocodecid",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.maudiocodecid = int(temp);
		}

		strValue = "";
		amf0::amf0Block5Value(block,"filesize",strValue);
		if (!strValue.empty())
		{
			temp = str2float(strValue.c_str());
			dInfo.mfilesize = int(temp);
		}

		amf0::amf0BlockRelease(block);
	}

	return dInfo;
}


//����PAT�İ�������Ϣ
void setPAT(int16 pmtPid,byte **ptr,int &PATlen)
{
	byte* &PATpack = *ptr;
	PATpack = new byte[188];
	PATlen = 0;
	byte section_len = 13;

	byte tableId = 0x00;                             //�̶�Ϊ0x00 ����־�Ǹñ���PAT
	byte sectionLength1 = (section_len >> 8) & 0x0f; //��ʾ����ֽں������õ��ֽ���������CRC32
	byte reserved1 = 0x03;                           // ����λ
	byte zero = 0;                                   //0
	byte sectionSyntaxIndicator = 1;                 //���﷨��־λ���̶�Ϊ1

	byte sectionLength2 = section_len & 0xff;		 //(section_length1 & 0x0F) << 8 | section_length2;

	byte transportStream_id1 = 0x00;
	byte transportStream_id2 = 0x01;				//�ô�������ID��������һ��������������·���õ���

	byte currentNextIndicator = 0x01; //���͵�PAT�ǵ�ǰ��Ч������һ��PAT��Ч
	byte versionNumber = 0x00;        //��Χ0-31����ʾPAT�İ汾��
	byte reserved2 = 0x03;            // ����λ

	byte sectionNumber = 0x00;     //�ֶεĺ��롣��һ��Ϊ00���Ժ�ÿ���ֶμ�1����������256���ֶ�
	byte lastSectionNumber = 0x00; //���һ���ֶεĺ���

	byte prograNumber1 = 0x00; //��Ŀ��
	byte programNumber2 = 0x01;

	byte programMapPID1 = byte(pmtPid>>8) & 0x1f; //��Ŀӳ����PID ��Ŀ��Ϊ0ʱ��Ӧ��PIDΪnetwork_PID
	byte reserved3 = 0x07;                        // ����λ

	byte programMapPID2 = byte(pmtPid & 0xff); // program_map_PID1 << 8 | program_map_PID2

	PATpack[0] = tableId;
	PATpack[1] = sectionSyntaxIndicator<<7 | zero<<6 | reserved1<<4 | sectionLength1;
	PATpack[2] = sectionLength2;
	PATpack[3] = transportStream_id1;
	PATpack[4] = transportStream_id2;
	PATpack[5] = reserved2<<6 | versionNumber<<1 | currentNextIndicator;
	PATpack[6] = sectionNumber;
	PATpack[7] = lastSectionNumber;
	PATpack[8] = prograNumber1;
	PATpack[9] = programNumber2;
	PATpack[10] = reserved3<<5 | programMapPID1;
	PATpack[11] = programMapPID2;

	uint32 crc32 = crc32n((unsigned char *)PATpack, 12);
	PATpack[12] = byte(crc32 >> 24);
	PATpack[13] = byte(crc32 >> 16);
	PATpack[14] = byte(crc32 >> 8);
	PATpack[15] = byte(crc32);
	PATlen = 16;
	/*
		//PAT:
		tableId                byte	//8
		SectionSyntaxIndicator byte	//1 ͨ����Ϊ��1��
		"0"                    byte //1
		Reserved               byte //2
		SectionLength          byte	//12
		transportStreamId      byte	//16
		Reserved               byte //2
		VersionNumber          byte	//5
		CurrentNextIndicator   byte	//1
		SectionNumber          byte	//8
		lastSectionNumber      byte	//8

		for(i=0;i<N;i++){
		    programNumber          	//16
		    Reserved                // 3
		    if(programNumber == 0){
		        networkPID         	//13
		    }
		    else{
		        programMapPID     	//13
		    }
		}
		CRCW32             			//32

	*/

}


//����PMT�İ�������Ϣ
void setPMT(int16 aPid,int16 vPid,int16 pcrPid,byte **ptr,int &PMTlen)
{
	bool VFlag = true;
	bool AFlag = true;
	//PMT:
	byte *&PMTpack = *ptr;
	PMTpack = new byte[188];
	PMTlen = 0;
	int sectionLen = 16 - 3;

	byte tableId = 0x02;									//0�̶�Ϊ0x02, ��ʾPMT��
	byte sectionLength1 = byte(sectionLen>>8) & 0x0f;	//1������λbit��Ϊ00����ָʾ�ε�byte�����ɸ���ʼ������CRC��
	byte reserved1 = 0x03;								//1 0x03
	byte zero = 0x00;									//1 0x01
	byte sectionSyntaxIndicator = 0x01;					//1 �̶�Ϊ0x01

	byte sectionLength2 = byte(sectionLen & 0xff); //2 (section_length1 & 0x0F) << 8 | section_length2;

	byte programNumber1 = 0x00; //3 ָ���ý�Ŀ��Ӧ�ڿ�Ӧ�õ�Program map PID
	byte programNumber2  = 0x01;

	byte currentNextIndicator = 0x01; //5 ����λ��1ʱ����ǰ���͵�Program map section���ã�
	byte versionNumber = 0x00;        //5 ָ��TS����Program map section�İ汾��
	byte reserved2 = 0x03;            //5 0x03

	byte sectionNumber = 0x00; //6 �̶�Ϊ0x00

	byte lastSectionNumber = 0x00; //7 �̶�Ϊ0x00

	byte pcrPID1 = byte(PCRpid>>8) & 0x1f; //8 ָ��TS����PIDֵ����TS������PCR��
	byte reserved3 = 0x07;                 //8 0x07

	byte pcrPID2 = byte(PCRpid & 0xff); //9 PCR_PID1 << 8 | PCR_PID2
	//��PCRֵ��Ӧ���ɽ�Ŀ��ָ���Ķ�Ӧ��Ŀ��
	//�������˽���������Ľ�Ŀ������PCR�޹أ�������ֵ��Ϊ0x1FFF��
	byte programInfoLength1 = 0x00; //10 ǰ��λbitΪ00������ָ���������Խ�Ŀ��Ϣ��������byte
	byte reserved4 = 0x0f;          //10 Ԥ��Ϊ0x0F

	byte programInfoLength2 = 0x00; //11 program_info_length1 <<8 | program_info_length2

	int Pos = 12;
	if (VFlag)
	{
		//��Ƶ��Ϣ
		byte videoStreamType = 0x1b; //17ָʾ�ض�PID�Ľ�ĿԪ�ذ������͡��ô�PID��elementary PIDָ��

		byte videoElementaryPID1 = byte((vPid >> 8) & 0x1f); //18����ָʾTS����PIDֵ����ЩTS��������صĽ�ĿԪ��
		byte videoReserved1 = 0x07;

		byte videoElementaryPID2 = byte(vPid & 0xff); //19 m_elementary_PID1 <<8 | m_elementary_PID2

		byte videoESnfoLength1 = 0x00; //20ǰ��λbitΪ00������ָʾ��������������ؽ�ĿԪ�ص�byte��
		byte videoReserved2 = 0x0f;

		byte videoESnfoLength2 = 0x00; //21 m_ES_nfo_length1 <<8 | m_ES_nfo_length2*/

		PMTpack[Pos] = videoStreamType;
		PMTpack[Pos+1] = videoReserved1<<5 | videoElementaryPID1;
		PMTpack[Pos+2] = videoElementaryPID2;
		PMTpack[Pos+3] = videoReserved2<<4 | videoESnfoLength1;
		PMTpack[Pos+4] = videoESnfoLength2;
		Pos += 5;
		sectionLen += 5;
	}
	if (AFlag)
	{
		//��Ƶ��Ϣ
		byte audioStreamType = 0x0f; //12ָʾ�ض�PID�Ľ�ĿԪ�ذ������͡��ô�PID��elementary PIDָ��

		byte audioElementaryPID1 = byte((aPid >> 8) & 0x1f); //13����ָʾTS����PIDֵ����ЩTS��������صĽ�ĿԪ��
		byte audioReserved1 = 0x07;

		byte audioElementaryPID2 = byte(aPid & 0xff); //14 m_elementary_PID1 <<8 | m_elementary_PID2

		byte audioESnfoLength1 = 0x00; //15ǰ��λbitΪ00������ָʾ��������������ؽ�ĿԪ�ص�byte��
		byte audioReserved2 = 0x0f;

		byte audioESnfoLength2 = 0x00; //16 m_ES_nfo_length1 <<8 | m_ES_nfo_length2

		PMTpack[Pos] = audioStreamType;
		PMTpack[Pos+1] = audioReserved1<<5 | audioElementaryPID1;
		PMTpack[Pos+2] = audioElementaryPID2;
		PMTpack[Pos+3] = audioReserved2<<4 | audioESnfoLength1;
		PMTpack[Pos+4] = audioESnfoLength2;
		Pos += 5;
		sectionLen += 5;
		//audio_descriptor;  //����
	}

	sectionLength2 = byte(sectionLen & 0xff);
	sectionLength1 = byte(sectionLen>>8) & 0x0f;

	PMTpack[0] = tableId;
	PMTpack[1] = sectionSyntaxIndicator<<7 | zero<<6 | reserved1<<4 | sectionLength1;
	PMTpack[2] = sectionLength2;
	PMTpack[3] = programNumber1;
	PMTpack[4] = programNumber2;
	PMTpack[5] = reserved2<<6 | versionNumber<<1 | currentNextIndicator;
	PMTpack[6] = sectionNumber;
	PMTpack[7] = lastSectionNumber;
	PMTpack[8] = reserved3<<5 | pcrPID1;
	PMTpack[9] = pcrPID2;
	PMTpack[10] = reserved4<<4 | programInfoLength1;
	PMTpack[11] = programInfoLength2;

	uint32 crc32 = crc32n((unsigned char *)PMTpack, Pos);
	PMTpack[Pos] = byte(crc32 >> 24);
	PMTpack[Pos+1] = byte(crc32 >> 16);
	PMTpack[Pos+2] = byte(crc32 >> 8);
	PMTpack[Pos+3] = byte(crc32);

	PMTlen = sectionLen + 3;
	/*
		table_id                //8
		Section_syntax_indicator//1 ͨ����Ϊ��1��
		"0"                     //1
		Reserved                //2
		Section_length          //12
		program_number          //16
		Reserved                //2
		Version_number          //5
		Current_next_indicator  //1
		Section_number          //8
		last_section_number     //8
		reserved                //3
		PCR_PID                 //13
		reserved                //4
		program_info_length     //12 ͷ��λΪ"00"
		for(i=0;i<N;i++){
		    descriptor()
		}
		for(i=0;i<N1;i++){
		    stream_type         //8
		    reserved            //3
		    elementary_PID      //13
		    reserved            //4
		   ES_info_length       //12 ͷ��λΪ"00"
		    for(j=0;j<N2;j++){
		        descriptor();
		    }
		}
		CRC_32                  //32 rpchof
	*/
}

void setSDT(byte **ptr, int &len)
{
	len = 188;
	*ptr = new byte[len];
	byte* &SDT = *ptr;
	memcpy(SDT,gSDT,len);
}

CSMux::CSMux()
{
	mheadFlag = false;
	minfoTag = false;
	memset(&mhead,0,sizeof(mhead));
	memset(&mhead,0,sizeof(mscript));
	memset(&mhead,0,sizeof(mcurTag));

	memset(mSpsPpsNal,0,sizeof(mSpsPpsNal));
	mSpsPpsNalLen = 0;

	memset(mheadBuf,0,sizeof(mheadBuf));
	mheadBufDL = 0;

	mTagSize = 0;
	memset(mTagSizeBuf,0,sizeof(mTagSizeBuf));
	mTagSizeBufDL = 0;

	mTagBuf = new byte[65536 * 2];
	mTagBufDLen	= 0;

	memset(mPAT,0,sizeof(mPAT));
	memset(mPMT,0,sizeof(mPMT));;

	memset(&mAudioSpecificConfig,0,sizeof(mAudioSpecificConfig));
	mAACFLag = false;

	mamcc = 0;
	mvmcc = 0;
	mpatmcc = 0;
	mpmtmcc = 0;
}

CSMux::~CSMux()
{
	delete[] mTagBuf;
}

void CSMux::init()
{
	mheadFlag = false;
	minfoTag = false;

	mheadBufDL = 0;
	mTagSize = -1;
	mTagSizeBufDL = 0;
	mTagBufDLen = 0;

	mamcc = 0;
	mvmcc = 0;
	mpatmcc = 0;
	mpmtmcc = 0;
}

void CSMux::release()
{
	mheadFlag = false;
	minfoTag = false;

	mheadBufDL = 0;
	mTagSize = -1;
	mTagSizeBufDL = 0;
	mTagBufDLen = 0;

	mamcc = 0;
	mvmcc = 0;
	mpatmcc = 0;
	mpmtmcc = 0;
}

//��һ��Tag�����ȴ��PES��
int  CSMux::packPES(byte *inBuf,int inLen,byte frameType,uint32 timestamp,byte **outBuf,int &outLen,uint64 &pts64)
{
	*outBuf = new byte[inLen+1024];
	outLen = 0;
	pts64 = 0;
	byte * &pesPack = *outBuf;
	int headLen = 0;	//����ͷ������
	int &packLen = outLen;	//����������
	uint64 pts = 0;
	uint64 dts = 0;

	uint64 cts = ((uint64)inBuf[2])<<16 | ((uint64)inBuf[3])<<8 | ((uint64)inBuf[4]); //ƫ����cts
	if (frameType == 'A')
	{
		cts = 0;
	}
	dts = ((uint64)timestamp) * 90;	//flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)
	pts = dts + cts*90;

	if (pts < dts)
	{
		pts = dts;
		logs->warn("*** [packPES] ERROR timestamp:  pts < dts ***");
	}
	//��ͷ����ʼ��ǰ׺��������ʶ��PES������Ϣ3���ֹ��ɡ�����ʼ��ǰ׺����23��������0����1����1�����ɵ�
	//3 0x000001 3�ֽڵ�ǰ׺
	//1	0xe0-0xef:v 0xc0-0xef:a
	//2 ����
	byte head[1024] = {0};
	head[0] = 0;
	head[1] = 0;
	head[2] = 1;
	head[3] = 0xE0; //pos= 3 ԭʼ�������ͺ���Ŀ��ȡֵ��1100��1111֮��
		//������ʶ��, ����������Ƶ����Ƶ����������
	head[4] = 0x00; //pos= 4 ��ʾ�Ӵ��ֽ�֮��PES����(��λ�ֽ�)��
	head[5] = 0x03;
	headLen += 6;

	byte PDflags = 0;
	//������ֵδȷ��
	if (frameType == 'I')
	{
		//I֡
		PDflags = 0x03;
	} 
	else if (frameType == 'P') 
	{
		//P֡
		PDflags = 0x03;
	} 
	else if (frameType == 'B') 
	{
		//B֡
		PDflags = 0x03;
	} 
	else if (frameType == 'A')
	{
		//��Ƶ
		head[3] = 0xC0;
		PDflags = 0x02;
	}
	//��ͷʶ���־��12������
	//head[6] = 0x80 //10000000
	byte fixBit = 0x02;				 //2 ����ֽ�
	byte PESscramblingControl = 0;   //2 PES��Ч���صļ���ģʽ��0��ʾ�����ܣ������ʾ�û��Զ��塣
	byte PESpriority = 0;            //1 PES���ݰ������ȼ�
	byte dataAlignmentIndicator = 0; //1 Ϊʱ�������˷���ͷ��֮��������������������ж���ķ��ʵ�Ԫ����
	byte copyright = 0;              //1 ��Ȩ����ʾ�а�Ȩ
	byte originalOrCopy = 0;         //1 pos= 6 1��ʾԭʼ���ݣ�0��ʾ����
	head[6] = fixBit<<6 | PESscramblingControl<<4 | PESpriority<<3 | dataAlignmentIndicator<<2 | copyright<<1 | originalOrCopy;

	byte PTSDTSflags = PDflags;      //2 10��ʾ����PTS�ֶΣ�11��ʾ����PTS��DTS�ֶΣ�00��ʾ������pts_dts��DTS��
	byte ESCRflag = 0;               //1 1��ʾESCR��PES�ײ����֣�0��ʾ������
	byte ESrateFlag = 0;             //1 1��ʾPES���麬��ES_rate�ֶΡ�0��ʾ�����С�
	byte DSMtrickModeFlag = 0;       //1 1��ʾ��λ��trick_mode_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�ֻ��DSM��Ч��
	byte additionalCopyInfoFlag = 0; //1 1��ʾ��copy_info_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�
	byte PESCRCflag = 0;             //1 1��ʾPES��������CRC�ֶΣ�0��ʾ�����ִ��ֶΡ�
	byte PESextensionFlag = 0;       //1 pos= 7 1��ʾ��չ�ֶ���PES��ͷ���ڣ�0��ʾ��չ�ֶβ�����
	/*if curStream.script.videodatarate > 0 {
		ESCRflag = 1
	}*/
	head[7] = PTSDTSflags<<6 | ESCRflag<<5 | ESrateFlag<<4 | DSMtrickModeFlag<<3 | additionalCopyInfoFlag<<2 | PESCRCflag<<1 | PESextensionFlag;
	head[8] = 0; // PESheaderDataLength = 0;         //��ʾ��ѡ�ֶκ�����ֶ���ռ���ֽ�����

	headLen += 3;

	if ((PTSDTSflags&0x02) > 0)
	{
		byte PTSbuf[5] = {0};
		//��ӣ��ĵ�˵�Ĳ��岻��
		PTSbuf[0] = (PTSDTSflags<<4) | (byte)((pts&0x1C0000000)>>29) | 0x01; //pts&0x 11000000 00000000 00000000 00000000
		PTSbuf[1] = (byte)((pts&0x3fc00000) >> 22);                          //pts&0x 00111111 11000000 00000000 00000000
		PTSbuf[2] = (byte)((pts&0x3f8000) >> 14) | 0x01;                     //pts&0x 00000000 00111111 10000000 00000000
		PTSbuf[3] = (byte)((pts&0x7f80) >> 7);                               //pts&0x 00000000 00000000 01111111 10000000
		PTSbuf[4] = (byte)(pts<<1) | 0x01;                                   //pts&0x 00000000 00000000 00000000 01111111

		head[8] += 5;
		memcpy(head+headLen, PTSbuf,5);
		headLen += 5;
		/*pts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|1|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		        pts1     pts2                            pts3
		*/
	}
	if ((PTSDTSflags&0x01) > 0) 
	{
		byte DTSbuf[5] = {0};
		DTSbuf[0] = (0x01<<4) | (byte)((dts&0x1C0000000)>>29) | 0x01;
		DTSbuf[1] = (byte)((dts&0x3fc00000) >> 22);
		DTSbuf[2] = (byte)((dts&0x3f8000)>>14) | 0x01;
		DTSbuf[3] = (byte)((dts&0x7f80) >> 7);
		DTSbuf[4] = (byte)(dts<<1) | 0x01;

		head[8] += 5;
		memcpy(head+headLen, DTSbuf,5);
		headLen += 5;
		/*dts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|0|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		         dts1    dts2                            dts3
		*/
	}

	if (ESCRflag > 0){

	}

	if (ESrateFlag > 0) {

	}

	if (ESrateFlag > 0) {

	}

	if (DSMtrickModeFlag > 0) {

	}

	if (additionalCopyInfoFlag > 0) {

	}

	if (PESextensionFlag > 0) {
		//������ʵ����һ��
	}
	if (frameType == 'I') 
	{ 
		//H264��Ƶ����
		//fmt.Printf("[PackPES] frameType == 'I'")
		byte *outNalBuf = NULL;
		int	 outNalLen = 0;
		parseNAL(inBuf, inLen,&outNalBuf,outNalLen);
		if (outNalLen <= 0)
		{ 
			//ΪSPS/PPS��Tag��������д��PES��(����I֡ǰ��I֡һ����)
			delete[] pesPack;
			pesPack = NULL;
			return 0;
		}

		packLen = headLen + mSpsPpsNalLen + 6 + outNalLen;
		head[4] = 0; // byte((packLen - 6) >> 8)
		head[5] = 0; // byte(packLen - 6)

		memcpy(pesPack, head,headLen);

		pesPack[headLen] = 0;
		pesPack[headLen+1] = 0;
		pesPack[headLen+2] = 0;
		pesPack[headLen+3] = 1;
		pesPack[headLen+4] = 9;
		pesPack[headLen+5] = 0xf0 ;                  //���sliceǰ����0x00 0x00 0x00 0x01 0x09 0xf0 ��ʼ��
		memcpy(pesPack+headLen+6, mSpsPpsNal,sizeof(mSpsPpsNal)); //���I֡ǰ���һ��Sps/Pps
		memcpy(pesPack+headLen+mSpsPpsNalLen+6, outNalBuf,outNalLen);
		delete[] outNalBuf;
		//Test(mSpsPpsNal[:], mSpsPpsNalLen)
		//Test(NALbuf[:], NALlen)
		//logs->debug(" [inLen packLen NALlen SpsPpsNalLen headLen %d %d %d %d %d] ", inLen, packLen, outNalLen, mSpsPpsNalLen, headLen);

	} 
	else if (frameType == 'A') 
	{
		//fmt.Printf("[PackPES] frameType == 'A'")
		if (inBuf[0]>>4 == 10) 
		{ 			
			//AAC
			byte *outAAC = NULL;
			int  outAACLen = 0;
			/*int ret = */parseAAC(inBuf, inLen,&outAAC,outAACLen);
			if (outAACLen > 0)
			{
				packLen = headLen + outAACLen;
				head[4] = (byte)((packLen - 6) >> 8);
				head[5] = (byte)(packLen - 6);
				memcpy(pesPack, head,headLen);
				memcpy(pesPack+headLen, outAAC,outAACLen);
				delete[] outAAC;
			}
			//fmt.Printf(" [inLen packLen %d %d] ", inLen, packLen)
		} 
		else 
		{
			//��AACĿǰ���򵥴���
			packLen = headLen + inLen - 1;
			head[4] = (byte)((packLen - 6) >> 8); //ȥ��6λͷ
			head[5] = (byte)(packLen - 6);
			memcpy(pesPack, head,headLen);
			memcpy(pesPack+headLen, inBuf+1,inLen-1);
		}
	} else {
		//fmt.Printf("[PackPES] frameType == else %c", frameType)
		byte *outNalBuf = NULL;
		int	 outNalLen = 0;
		parseNAL(inBuf, inLen,&outNalBuf,outNalLen);
		packLen = headLen + outNalLen + 6;
		head[4] = 0 ;// byte((packLen - 6) >> 8)
		head[5] = 0; // byte(packLen - 6)

		memcpy(pesPack, head,headLen);
		pesPack[headLen] = 0;
		pesPack[headLen+1] = 0;
		pesPack[headLen+2] = 0;
		pesPack[headLen+3] = 1;
		pesPack[headLen+4] = 9;
		pesPack[headLen+5] = 0xf0; //���sliceǰ����0x00 0x00 0x00 0x01 0x09 0xf0 ��ʼ��
		memcpy(pesPack+headLen+6, outNalBuf,outNalLen);
		delete[] outNalBuf;
		//fmt.Printf(" [inLen packLen NALlen headLen %d %d %d %d] ", inLen, packLen, NALlen, headLen)
		//Test(NALbuf[:], NALlen)
	}
	pts64 = pts;
	return 1;
}

int	 CSMux::packTS(byte *inBuf,int inLen,byte frameType,byte pusi,byte afc,
				   byte *mcc,int16 pid,uint32 timestamp,byte **outBuf,int &outLen)
{
	int max = (int)(inLen*110/18800)*188 + 188*2;
	*outBuf = new byte[max];
	byte *&TSstream = *outBuf;
	outLen = 0;
	int &streamLen = outLen;
	bool StuffFlag = false;
	int StuffLen = 0;
	/*if inLen < 0 {
		fmt.Println("[PackTS] ERROR inLen")
		fmt.Println(inLen)
		return TSstream[:], -1
	}*/

// 	if len(inBuf) < inLen {
// 		fmt.Println("[PackTS] ERROR inLen :len(inBuf),inLen ", len(inBuf), inLen)
// 		inLen = len(inBuf)
// 	}

	uint64 DTS = uint64(timestamp) * 90; //flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)

	int dealPos = 0;
	if (inLen > 65536*4) 
	{
		logs->warn("*** [PackTS] too long inLen ***");
	}
	//��ȥPSI�Ĵ��
	for (dealPos = 0; dealPos < inLen; ) 
	{
		byte TSpack[188] = {0};
		int headlen = 0;
		byte head[4] = {0};
		byte adaptionFiled[184] = {0};
		int adaptationFiledLength = 0;

		byte synByte = 0x47;                   //ͬ���ֽ�, �̶�Ϊ,��ʾ�������һ��TS����
		byte transportErrorIndicator = 0;      //��������ָʾ��
		byte payloadUnitStartIndicator = pusi; //��Ч���ص�Ԫ��ʼָʾ��, 1����Ϊ������ʼ��
		byte transport_priority = 0;           //��������, 1��ʾ�����ȼ�,������ƿ����õ��������ò���
		int16 PID = pid;                       //PID = 0x100

		byte transportScramblingControl = 0; //������ſ���(����)
		byte adaptionFieldControl = afc;     //����Ӧ���� 01������Ч���أ�10���������ֶΣ�11����
		byte continuityCounter = *mcc;       //���������� һ��4bit�ļ���������Χ0-15
		head[0] = synByte;
		head[1] = transportErrorIndicator<<7 | payloadUnitStartIndicator<<6 | transport_priority<<5 | byte((PID<<3)>>11);
		head[2] = (byte)(PID);
		head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		headlen += 4;

		if (inLen-dealPos < 182 && pusi == 1) 
		{ 
			//���ݲ�������Ҫ�����188�ֽ�
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
			StuffFlag = true;
		} 
		else if (inLen-dealPos < 184 && pusi == 0) 
		{		
			//�ְ�(û������Ӧ����1B����־λ1B)
// 			if (pusi == 0) 
			{
				adaptionFieldControl = 0x03;
				head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
				StuffFlag = true;
			}
		}
		if ((frameType == 'I' || frameType == 'A') && pusi == 1) 
		{ 
			//����ԭʼ���ڵ�
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		}
		if (adaptionFieldControl > 1)
		{
			byte discontinuityIndicator = 0; //������״ָ̬ʾ
			byte randomAccessIndicator = 0;  //
			if ((frameType == 'I' || frameType == 'A') && pusi == 1)
			{
				randomAccessIndicator = 1;
			}
			byte elementaryStreamPriorityIndicator = 0; //Ϊ1��ǰ�����нϸ����ȼ�,�ж�������ڻ��õ������Բ����ǣ�
			byte pcrFlag = 0 ;                          //pcr��־   pes����ǰ�� ��������
			if (pusi == 1 && frameType != 'A')
			{
				pcrFlag = 1;
			}
			byte opcrFlag = 0;                 //��������
			byte splicingPointFlag = 0;        //ƴ�ӵ㣨ò��û�ã�
			byte transportPrivateDataFlag = 0; //˽�������ֽڣ����Բ����ǣ�
			byte adaptationFiledExtension = 0; //��չ����Ӧ�ֶ�
			adaptionFiled[1] = discontinuityIndicator<<7 | randomAccessIndicator<<6 | elementaryStreamPriorityIndicator<<5 | pcrFlag<<4 | opcrFlag<<3 | splicingPointFlag<<2 | transportPrivateDataFlag<<1 | adaptationFiledExtension;
			adaptationFiledLength += 2;
			if (pcrFlag > 0)
			{
				byte *outPcrbuf = NULL;
				int  outPcrBufLen = 0;
				setPcr(DTS,&outPcrbuf,outPcrBufLen);
				memcpy(adaptionFiled+adaptationFiledLength, outPcrbuf,outPcrBufLen);
				adaptationFiledLength += 6;
				delete[] outPcrbuf;
				// 48 bit
			}
			if (opcrFlag > 0)
			{
				// 48 bit
			}
			if (splicingPointFlag > 0)
			{
				// 8 bit
			}
			if (transportPrivateDataFlag > 0)
			{
				byte transportPrivateDataLen = 0; // 8 bit
				if (transportPrivateDataLen > 0)
				{
					transportPrivateDataLen += 0;
				}

			}
			if (adaptationFiledExtension > 0)
			{
				//����Ҳ����һ��
				byte adaptationExFiledLength = 0;
				if (false)
				{
					adaptationExFiledLength++;
				}
			}

			if (StuffFlag)
			{
				StuffLen = 184 - (inLen - dealPos) - 2;
				for (int i = adaptationFiledLength; StuffLen > 0 && i < StuffLen+adaptationFiledLength; i++)
				{
					adaptionFiled[i] = 0xFF;
				}
				adaptationFiledLength += StuffLen;
			}
			adaptionFiled[0] = (byte)(adaptationFiledLength - 1);
			memcpy(TSpack+4, adaptionFiled,adaptationFiledLength);
			headlen += (int)(adaptationFiledLength);
		}
		*mcc = (*mcc + 1) % 16;
		/*TSpack[headlen] = 0
		headlen++
		copy(TSpack[:], head[:])
		if pusi == 1 {
			TSpack[headlen] = 0
			headlen++
		}*/
		memcpy(TSpack, head,sizeof(head));
		if (dealPos+188-headlen <= inLen) 
		{
			memcpy(TSpack+headlen, inBuf+dealPos,188-headlen);
		}
		else 
		{
			//logs->debug(" [PackTS] 901 dealPos+188-headlen=%d, inLen=%d ", dealPos+188-headlen, inLen);
			memcpy(TSpack+headlen, inBuf+dealPos,inLen-dealPos);
		}

		if (streamLen+188 > max) 
		{
			logs->warn(" [PackTS] 916 too big max=%d, inLen=%d, streamLen+188=%d ", max, inLen, streamLen+188);
		}
		dealPos = dealPos + 188 - headlen;
		memcpy(TSstream+streamLen, TSpack,sizeof(TSpack));
		streamLen = streamLen + 188;
		pusi = 0;
	}
	//fmt.Printf(" streamLen [TSstream[streamLen-1]  inBuf[inLen-1]  %d %d %d] ", streamLen, TSstream[streamLen-1], inBuf[inLen-1])
	return 1;
}

int  CSMux::packPES(byte *inBuf,int inLen,byte frameType,uint32 timestamp,byte pusi,byte afc,
			 byte *mcc,int16 pid,TsChunkArray **tca,uint64 &pts64,int pesLen)
{
	pts64 = 0;
	int handleLen = 0;	//����ͷ������
	int packLen = 0;	//����������
	uint64 pts = 0;
	uint64 dts = 0;

	uint64 cts = ((uint64)inBuf[2])<<16 | ((uint64)inBuf[3])<<8 | ((uint64)inBuf[4]); //ƫ����cts
	if (frameType == 'A')
	{
		cts = 0;
	}
	dts = ((uint64)timestamp) * 90;	//flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)
	pts = dts + cts*90;

	if (pts < dts)
	{
		pts = dts;
		logs->warn("*** [packPES] ERROR timestamp:  pts < dts ***");
	}
	//��ͷ����ʼ��ǰ׺��������ʶ��PES������Ϣ3���ֹ��ɡ�����ʼ��ǰ׺����23��������0����1����1�����ɵ�
	//3 0x000001 3�ֽڵ�ǰ׺
	//1	0xe0-0xef:v 0xc0-0xef:a
	//2 ����
	byte head[50] = {0};
	head[0] = 0;
	head[1] = 0;
	head[2] = 1;
	head[3] = 0xE0; //pos= 3 ԭʼ�������ͺ���Ŀ��ȡֵ��1100��1111֮��
		//������ʶ��, ����������Ƶ����Ƶ����������
	head[4] = 0x00; //pos= 4 ��ʾ�Ӵ��ֽ�֮��PES����(��λ�ֽ�)��
	head[5] = 0x03;
	handleLen += 6;

	byte PDflags = 0;
	//������ֵδȷ��
	if (frameType == 'I')
	{
		//I֡
		PDflags = 0x03;
	} 
	else if (frameType == 'P') 
	{
		//P֡
		PDflags = 0x03;
	} 
	else if (frameType == 'B') 
	{
		//B֡
		PDflags = 0x03;
	} 
	else if (frameType == 'A')
	{
		//��Ƶ
		head[3] = 0xC0;
		PDflags = 0x02;
	}
	//��ͷʶ���־��12������
	//head[6] = 0x80 //10000000
	byte fixBit = 0x02;				 //2 ����ֽ�
	byte PESscramblingControl = 0;   //2 PES��Ч���صļ���ģʽ��0��ʾ�����ܣ������ʾ�û��Զ��塣
	byte PESpriority = 0;            //1 PES���ݰ������ȼ�
	byte dataAlignmentIndicator = 0; //1 Ϊʱ�������˷���ͷ��֮��������������������ж���ķ��ʵ�Ԫ����
	byte copyright = 0;              //1 ��Ȩ����ʾ�а�Ȩ
	byte originalOrCopy = 0;         //1 pos= 6 1��ʾԭʼ���ݣ�0��ʾ����
	head[6] = fixBit<<6 | PESscramblingControl<<4 | PESpriority<<3 | dataAlignmentIndicator<<2 | copyright<<1 | originalOrCopy;

	byte PTSDTSflags = PDflags;      //2 10��ʾ����PTS�ֶΣ�11��ʾ����PTS��DTS�ֶΣ�00��ʾ������pts_dts��DTS��
	byte ESCRflag = 0;               //1 1��ʾESCR��PES�ײ����֣�0��ʾ������
	byte ESrateFlag = 0;             //1 1��ʾPES���麬��ES_rate�ֶΡ�0��ʾ�����С�
	byte DSMtrickModeFlag = 0;       //1 1��ʾ��λ��trick_mode_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�ֻ��DSM��Ч��
	byte additionalCopyInfoFlag = 0; //1 1��ʾ��copy_info_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�
	byte PESCRCflag = 0;             //1 1��ʾPES��������CRC�ֶΣ�0��ʾ�����ִ��ֶΡ�
	byte PESextensionFlag = 0;       //1 pos= 7 1��ʾ��չ�ֶ���PES��ͷ���ڣ�0��ʾ��չ�ֶβ�����
	/*if curStream.script.videodatarate > 0 {
		ESCRflag = 1
	}*/
	head[7] = PTSDTSflags<<6 | ESCRflag<<5 | ESrateFlag<<4 | DSMtrickModeFlag<<3 | additionalCopyInfoFlag<<2 | PESCRCflag<<1 | PESextensionFlag;
	head[8] = 0; // PESheaderDataLength = 0;         //��ʾ��ѡ�ֶκ�����ֶ���ռ���ֽ�����

	handleLen += 3;

	byte PTSbuf[5] = {0};
	int  PTSbufLen = 0;
	if ((PTSDTSflags&0x02) > 0)
	{
		//��ӣ��ĵ�˵�Ĳ��岻��
		PTSbuf[0] = (PTSDTSflags<<4) | (byte)((pts&0x1C0000000)>>29) | 0x01; //pts&0x 11000000 00000000 00000000 00000000
		PTSbuf[1] = (byte)((pts&0x3fc00000) >> 22);                          //pts&0x 00111111 11000000 00000000 00000000
		PTSbuf[2] = (byte)((pts&0x3f8000) >> 14) | 0x01;                     //pts&0x 00000000 00111111 10000000 00000000
		PTSbuf[3] = (byte)((pts&0x7f80) >> 7);                               //pts&0x 00000000 00000000 01111111 10000000
		PTSbuf[4] = (byte)(pts<<1) | 0x01;                                   //pts&0x 00000000 00000000 00000000 01111111

		head[8] += 5;
		memcpy(head+handleLen, PTSbuf,5);
		handleLen += 5;
		PTSbufLen = 5;
		/*pts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|1|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		        pts1     pts2                            pts3
		*/
	}
	byte DTSbuf[5] = {0};
	int  DTSbufLen = 0;
	if ((PTSDTSflags&0x01) > 0) 
	{
		
		DTSbuf[0] = (0x01<<4) | (byte)((dts&0x1C0000000)>>29) | 0x01;
		DTSbuf[1] = (byte)((dts&0x3fc00000) >> 22);
		DTSbuf[2] = (byte)((dts&0x3f8000)>>14) | 0x01;
		DTSbuf[3] = (byte)((dts&0x7f80) >> 7);
		DTSbuf[4] = (byte)(dts<<1) | 0x01;

		head[8] += 5;
		memcpy(head+handleLen, DTSbuf,5);
		handleLen += 5;
		DTSbufLen = 5;
		/*dts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|0|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		         dts1    dts2                            dts3
		*/
	}

	if (ESCRflag > 0){

	}

	if (ESrateFlag > 0) {

	}

	if (ESrateFlag > 0) {

	}

	if (DSMtrickModeFlag > 0) {

	}

	if (additionalCopyInfoFlag > 0) {

	}

	if (PESextensionFlag > 0) {
		//������ʵ����һ��
	}
	if (frameType == 'I') 
	{
// 		printf(">>>>>>>>>>>>handle video\n");
		//H264��Ƶ����
		//fmt.Printf("[PackPES] frameType == 'I'")
// 		byte *outNalBuf = NULL;
// 		int	 outNalLen = 0;
// 		parseNAL(inBuf, inLen,&outNalBuf,outNalLen);
// 		if (outNalLen <= 0)
// 		{ 
// 			//ΪSPS/PPS��Tag��������д��PES��(����I֡ǰ��I֡һ����)
// 			return 0;
// 		}

// 		packLen = handleLen + mSpsPpsNalLen + 6 + outNalLen;
		head[4] = 0; // byte((packLen - 6) >> 8)
		head[5] = 0; // byte(packLen - 6)
// 		printf("============ packet len=%d,pesLen=%d ===============\n",packLen,pesLen);
		writeES((char *)head,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
// 		printf("============ head len=%d ===============\n",handleLen);
		byte pesPack[6] = {0};
		pesPack[0] = 0;
		pesPack[1] = 0;
		pesPack[2] = 0;
		pesPack[3] = 1;
		pesPack[4] = 9;
		pesPack[5] = 0xf0 ;                  //���sliceǰ����0x00 0x00 0x00 0x01 0x09 0xf0 ��ʼ��

		handleLen = 6;
		writeES((char *)pesPack,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
// 		printf("============ pesPack len=%d ===============\n",handleLen);

		writeES((char *)mSpsPpsNal,mSpsPpsNalLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
// 		printf("============ mSpsPpsNal len=%d ===============\n",mSpsPpsNalLen);
		
		//writeES((char *)outNalBuf,outNalLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		parseNALEx(inBuf, inLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
// 		printf("============ outNalBuf len=%d ===============\n",outNalLen);

//		delete[] outNalBuf;
		//Test(mSpsPpsNal[:], mSpsPpsNalLen)
		//Test(NALbuf[:], NALlen)
		//logs->debug(" [inLen packLen NALlen SpsPpsNalLen headLen %d %d %d %d %d] ", inLen, packLen, outNalLen, mSpsPpsNalLen, headLen);
		assert(pesLen == 0);
	} 
	else if (frameType == 'A') 
	{
		//fmt.Printf("[PackPES] frameType == 'A'")
		if (inBuf[0]>>4 == 10) 
		{ 			
			//AAC
// 			byte *outAAC = NULL;
// 			int  outAACLen = 0;
// 			/*int ret = */parseAAC(inBuf, inLen,&outAAC,outAACLen);
			int  outAACLen = parseAACLen(inBuf, inLen);
			if (outAACLen > 0)
			{
// 				printf(">>>>>>>>>>>>handle audio\n");
				packLen = handleLen + outAACLen;
				head[4] = (byte)((packLen - 6) >> 8);
				head[5] = (byte)(packLen - 6);
				writeES((char *)head,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);

				//writeES((char *)outAAC,outAACLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
				//delete[] outAAC;
				parseAACEx(inBuf,inLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
				assert(pesLen == 0);
			}
			//fmt.Printf(" [inLen packLen %d %d] ", inLen, packLen)
		} 
		else 
		{
			//��AACĿǰ���򵥴���
			packLen = handleLen + inLen - 1;
			head[4] = (byte)((packLen - 6) >> 8);
			head[5] = (byte)(packLen - 6);
			writeES((char *)head,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);

			writeES((char *)(inBuf+1),inLen-1,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		}
	} 
	else 
	{
// 		printf(">>>>>>>>>>>>handle video\n");
		//fmt.Printf("[PackPES] frameType == else %c", frameType)
// 		byte *outNalBuf = NULL;
// 		int	 outNalLen = 0;
// 		parseNAL(inBuf, inLen,&outNalBuf,outNalLen);
// 		packLen = handleLen + outNalLen + 6;
		head[4] = 0 ;// byte((packLen - 6) >> 8)
		head[5] = 0; // byte(packLen - 6)

		writeES((char *)head,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);

		byte pesPack[6] = {0};
		pesPack[0] = 0;
		pesPack[1] = 0;
		pesPack[2] = 0;
		pesPack[3] = 1;
		pesPack[4] = 9;
		pesPack[5] = 0xf0 ;                  //���sliceǰ����0x00 0x00 0x00 0x01 0x09 0xf0 ��ʼ��

		handleLen = 6;
		writeES((char *)pesPack,handleLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);

		//writeES((char *)outNalBuf,outNalLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		//delete[] outNalBuf;
		parseNALEx(inBuf, inLen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		//fmt.Printf(" [inLen packLen NALlen headLen %d %d %d %d] ", inLen, packLen, NALlen, headLen)
		//Test(NALbuf[:], NALlen)
	}
	pts64 = pts;
	return 1;
}

int  CSMux::packTS(byte *inBuf,int inLen,byte frameType,uint32 timestamp,byte pusi,byte afc,
			byte *mcc,int16 pid,TsChunkArray **tca,uint64 &pts64)
{
	int pesLen = packPESLen(inBuf,inLen,frameType,timestamp);
	TsChunkArray *&chunk = *tca;
	if (chunk == NULL)
	{
		//������ֵδȷ��
		if (frameType == 'I' || frameType == 'P' || frameType == 'B')
		{
			chunk = allocTsChunkArray(TS_VIDEO_SLICE_LEN);
		} 
		else if (frameType == 'A')
		{
			chunk = allocTsChunkArray(TS_AUDIO_SLICE_LEN);
		}		
	}
	bool StuffFlag = false;
	int  StuffLen = 0;
	uint64 DTS = uint64(timestamp) * 90; //flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)

	int headlen = 0;
	byte head[4] = {0};
	byte adaptionFiled[2] = {0};
	int adaptationFiledLength = 0;

	byte synByte = 0x47;                   //ͬ���ֽ�, �̶�Ϊ,��ʾ�������һ��TS����
	byte transportErrorIndicator = 0;      //��������ָʾ��
	byte payloadUnitStartIndicator = pusi; //��Ч���ص�Ԫ��ʼָʾ��, 1����Ϊ������ʼ��
	byte transport_priority = 0;           //��������, 1��ʾ�����ȼ�,������ƿ����õ��������ò���
	int16 PID = pid;                       //PID = 0x100

	byte transportScramblingControl = 0; //������ſ���(����)
	byte adaptionFieldControl = afc;     //����Ӧ���� 01������Ч���أ�10���������ֶΣ�11����
	byte continuityCounter = *mcc;       //���������� һ��4bit�ļ���������Χ0-15
	head[0] = synByte;
	head[1] = transportErrorIndicator<<7 | payloadUnitStartIndicator<<6 | transport_priority<<5 | byte((PID<<3)>>11);
	head[2] = (byte)(PID);
	head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
	headlen += 4;

	if (pesLen < 182 && pusi == 1) 
	{ 
		//���ݲ�������Ҫ�����188�ֽ�
		adaptionFieldControl = 0x03;
		head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		StuffFlag = true;
	} 
	else if (pesLen < 184) 
	{		
		//�ְ�(û������Ӧ����1B����־λ1B)
		if (pusi == 0) 
		{
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
			StuffFlag = true;
		} 
		else if (pusi == 1 && frameType != 'I' && frameType != 'A') 
		{
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
			StuffFlag = true;
		}		
	}
	if ((frameType == 'I' || frameType == 'A') && pusi == 1) //ֻ��Ҫ���б�����Ϣ��֡����Ҫ�����չ��Ϣ
	{ 
		//����ԭʼ���ڵ�
		adaptionFieldControl = 0x03;
		head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		if (pesLen < 184)
		{
			StuffFlag = true;
		}
	}
	int writeLen = 0;
	writeChunk(NULL,0,chunk,(char *)head,4,writeLen);
// 	printf("1 writeChunk size %d\n",4);
	if (adaptionFieldControl > 1)
	{
		byte discontinuityIndicator = 0; //������״ָ̬ʾ
		byte randomAccessIndicator = 0;  //
		if ((frameType == 'I' || frameType == 'A') && pusi == 1)
		{
			randomAccessIndicator = 1;
		}
		byte elementaryStreamPriorityIndicator = 0; //Ϊ1��ǰ�����нϸ����ȼ�,�ж�������ڻ��õ������Բ����ǣ�
		byte pcrFlag = 0 ;                          //pcr��־   pes����ǰ�� ��������
		if (pusi == 1 && frameType != 'A')
		{
			pcrFlag = 1;
		}
		byte opcrFlag = 0;                 //��������
		byte splicingPointFlag = 0;        //ƴ�ӵ㣨ò��û�ã�
		byte transportPrivateDataFlag = 0; //˽�������ֽڣ����Բ����ǣ�
		byte adaptationFiledExtension = 0; //��չ����Ӧ�ֶ�
		adaptionFiled[1] = discontinuityIndicator<<7 | randomAccessIndicator<<6 | elementaryStreamPriorityIndicator<<5 | pcrFlag<<4 | opcrFlag<<3 | splicingPointFlag<<2 | transportPrivateDataFlag<<1 | adaptationFiledExtension;
		//���� adaptationFiledLength
		adaptationFiledLength += 2;
		if (pcrFlag > 0)
		{
			adaptationFiledLength += 6;
			// 48 bit
		}
		if (opcrFlag > 0)
		{
			// 48 bit
		}
		if (splicingPointFlag > 0)
		{
			// 8 bit
		}
		if (transportPrivateDataFlag > 0)
		{
			byte transportPrivateDataLen = 0; // 8 bit
			if (transportPrivateDataLen > 0)
			{
				transportPrivateDataLen += 0;
			}

		}
		if (adaptationFiledExtension > 0)
		{
			//����Ҳ����һ��
			byte adaptationExFiledLength = 0;
			if (false)
			{
				adaptationExFiledLength++;
			}
		}

		if (StuffFlag)
		{
			StuffLen = 184 - (pesLen) - 2;
			adaptationFiledLength += StuffLen;
		}
		adaptionFiled[0] = (byte)(adaptationFiledLength - 1);
		if (adaptationFiledLength <= 1)
		{
			writeChunk(NULL,0,chunk,(char *)adaptionFiled,1,writeLen);
// 			printf("22 writeChunk size %d\n",1);
		}
		else
		{
			writeChunk(NULL,0,chunk,(char *)adaptionFiled,2,writeLen);
// 			printf("2 writeChunk size %d\n",2);
		}
		
		//���� adaptationFiledLength ����
		if (pcrFlag > 0)
		{
			byte *outPcrbuf = NULL;
			int  outPcrBufLen = 0;
			setPcr(DTS,&outPcrbuf,outPcrBufLen);
			writeChunk(NULL,0,chunk,(char *)outPcrbuf,outPcrBufLen,writeLen);
// 			printf("3 writeChunk size %d\n",outPcrBufLen);
			delete[] outPcrbuf;
			// 48 bit
		}
		if (opcrFlag > 0)
		{
			// 48 bit
		}
		if (splicingPointFlag > 0)
		{
			// 8 bit
		}
		if (transportPrivateDataFlag > 0)
		{
			byte transportPrivateDataLen = 0; // 8 bit
			if (transportPrivateDataLen > 0)
			{
				transportPrivateDataLen += 0;
			}

		}
		if (adaptationFiledExtension > 0)
		{
			//����Ҳ����һ��
			byte adaptationExFiledLength = 0;
			if (false)
			{
				adaptationExFiledLength++;
			}
		}

		if (StuffFlag)
		{
			char ch = 0xFF;
			for (int i = 0; StuffLen > 0 && i < StuffLen; i++)
			{
				writeChunk(NULL,0,chunk,(char *)&ch,1,writeLen);
			}
// 			printf("4 writeChunk size %d\n",StuffLen);
		}
	}
	return packPES(inBuf,inLen,frameType,timestamp,0,afc,mcc,pid,tca,pts64,pesLen);
}

int  CSMux::writeTsHeader(byte frameType,uint32 timestamp,byte pusi,byte afc,
				   byte *mcc,int16 pid,TsChunkArray **tca,int pesLen)
{
	*mcc = (*mcc + 1) % 16;
	TsChunkArray *&chunk = *tca;
	if (chunk == NULL)
	{
		//������ֵδȷ��
		if (frameType == 'I' || frameType == 'P' || frameType == 'B')
		{
			chunk = allocTsChunkArray(TS_VIDEO_SLICE_LEN);
		} 
		else if (frameType == 'A')
		{
			chunk = allocTsChunkArray(TS_AUDIO_SLICE_LEN);
		}		
	}
	bool StuffFlag = false;
	int  StuffLen = 0;
	uint64 DTS = uint64(timestamp) * 90; //flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)

	int headlen = 0;
	byte head[4] = {0};
	byte adaptionFiled[2] = {0};
	int adaptationFiledLength = 0;

	byte synByte = 0x47;                   //ͬ���ֽ�, �̶�Ϊ,��ʾ�������һ��TS����
	byte transportErrorIndicator = 0;      //��������ָʾ��
	byte payloadUnitStartIndicator = 0; //��Ч���ص�Ԫ��ʼָʾ��, 1����Ϊ������ʼ��
	byte transport_priority = 0;           //��������, 1��ʾ�����ȼ�,������ƿ����õ��������ò���
	int16 PID = pid;                       //PID = 0x100

	byte transportScramblingControl = 0; //������ſ���(����)
	byte adaptionFieldControl = afc;     //����Ӧ���� 01������Ч���أ�10���������ֶΣ�11����
	byte continuityCounter = *mcc;       //���������� һ��4bit�ļ���������Χ0-15
	head[0] = synByte;
	head[1] = transportErrorIndicator<<7 | payloadUnitStartIndicator<<6 | transport_priority<<5 | byte((PID<<3)>>11);
	head[2] = (byte)(PID);
	head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
	headlen += 4;

	if (pesLen < 182 && pusi == 1) 
	{ 
		//���ݲ�������Ҫ�����188�ֽ�
		adaptionFieldControl = 0x03;
		head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		StuffFlag = true;
	} 
	else if (pesLen < 184) 
	{		
		//�ְ�(û������Ӧ����1B����־λ1B)
		if (pusi == 0) 
		{
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
			StuffFlag = true;
		} 
		else if (pusi == 1 && frameType != 'I' && frameType != 'A') 
		{
			adaptionFieldControl = 0x03;
			head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
			StuffFlag = true;
		}		
	}
	if ((frameType == 'I' || frameType == 'A') && pusi == 1) 
	{ 
		//����ԭʼ���ڵ�
		adaptionFieldControl = 0x03;
		head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;
		if (pesLen < 184)
		{
			StuffFlag = true;
		}
	}
	int writeLen = 0;
	writeChunk(NULL,0,chunk,(char *)head,4,writeLen);
// 	printf("5 writeChunk size %d\n",4);

	if (adaptionFieldControl > 1)
	{
		byte discontinuityIndicator = 0; //������״ָ̬ʾ
		byte randomAccessIndicator = 0;  //
		if ((frameType == 'I' || frameType == 'A') && pusi == 1)
		{
			randomAccessIndicator = 1;
		}
		byte elementaryStreamPriorityIndicator = 0; //Ϊ1��ǰ�����нϸ����ȼ�,�ж�������ڻ��õ������Բ����ǣ�
		byte pcrFlag = 0 ;                          //pcr��־   pes����ǰ�� ��������
		if (pusi == 1 && frameType == 'I')
		{
			pcrFlag = 1;
		}
		byte opcrFlag = 0;                 //��������
		byte splicingPointFlag = 0;        //ƴ�ӵ㣨ò��û�ã�
		byte transportPrivateDataFlag = 0; //˽�������ֽڣ����Բ����ǣ�
		byte adaptationFiledExtension = 0; //��չ����Ӧ�ֶ�
		adaptionFiled[1] = discontinuityIndicator<<7 | randomAccessIndicator<<6 | elementaryStreamPriorityIndicator<<5 | pcrFlag<<4 | opcrFlag<<3 | splicingPointFlag<<2 | transportPrivateDataFlag<<1 | adaptationFiledExtension;
		//���� adaptationFiledLength
		adaptationFiledLength += 2;
		if (pcrFlag > 0)
		{
			adaptationFiledLength += 6;
			// 48 bit
		}
		if (opcrFlag > 0)
		{
			// 48 bit
		}
		if (splicingPointFlag > 0)
		{
			// 8 bit
		}
		if (transportPrivateDataFlag > 0)
		{
			byte transportPrivateDataLen = 0; // 8 bit
			if (transportPrivateDataLen > 0)
			{
				transportPrivateDataLen += 0;
			}

		}
		if (adaptationFiledExtension > 0)
		{
			//����Ҳ����һ��
			byte adaptationExFiledLength = 0;
			if (false)
			{
				adaptationExFiledLength++;
			}
		}

		if (StuffFlag)
		{
			StuffLen = 184 - (pesLen) - 2;
			adaptationFiledLength += StuffLen;
		}
		adaptionFiled[0] = (byte)(adaptationFiledLength - 1);
		if (adaptationFiledLength <= 1)
		{
			writeChunk(NULL,0,chunk,(char *)adaptionFiled,1,writeLen);
// 			printf("77 writeChunk size %d\n",1);
		}
		else
		{
			writeChunk(NULL,0,chunk,(char *)adaptionFiled,2,writeLen);
// 			printf("7 writeChunk size %d\n",2);
		}
		//���� adaptationFiledLength ����
		if (pcrFlag > 0)
		{
			byte *outPcrbuf = NULL;
			int  outPcrBufLen = 0;
			setPcr(DTS,&outPcrbuf,outPcrBufLen);
			writeChunk(NULL,0,chunk,(char *)outPcrbuf,outPcrBufLen,writeLen);
			delete[] outPcrbuf;
// 			printf("8 writeChunk size %d\n",outPcrBufLen);
			// 48 bit
		}
		if (opcrFlag > 0)
		{
			// 48 bit
		}
		if (splicingPointFlag > 0)
		{
			// 8 bit
		}
		if (transportPrivateDataFlag > 0)
		{
			byte transportPrivateDataLen = 0; // 8 bit
			if (transportPrivateDataLen > 0)
			{
				transportPrivateDataLen += 0;
			}

		}
		if (adaptationFiledExtension > 0)
		{
			//����Ҳ����һ��
			byte adaptationExFiledLength = 0;
			if (false)
			{
				adaptationExFiledLength++;
			}
		}

		if (StuffFlag)
		{
			char ch = 0xFF;
			for (int i = 0; StuffLen > 0 && i < StuffLen; i++)
			{
				writeChunk(NULL,0,chunk,(char *)&ch,1,writeLen);
			}
// 			printf("9 writeChunk size %d,pesLen %d\n",StuffLen,pesLen);
		}
	}	
	return 1;
}

void  CSMux::writeES(char *inBuf,int inLen,byte frameType,uint32 timestamp,byte pusi,byte afc,
			 byte *mcc,int16 pid,TsChunkArray **tca,uint64 &pts64,int &pesLen)
{
// 	printf("12 writeES inLen %d,pesLen %d\n",inLen,pesLen);
	int writeLen = 0;
	int ret = 0;
	int handleLen = 0;
	while (inLen > 0)
	{
		writeLen = 0;		
		ret = writeChunk(NULL,0,*tca,inBuf+handleLen,inLen,writeLen);
// 		printf("10 writeChunk size %d,inLen %d,tca.msliceSize %d\n",writeLen,inLen,(*tca)->msliceSize);
		if (ret == 1)
		{
			pesLen -= writeLen;
			inLen -= writeLen;
			handleLen += writeLen;
			if (pesLen > 0)
			{
// 				printf("11 pesLen %d\n",pesLen);
				writeTsHeader(frameType,timestamp,0,afc,mcc,pid,tca,pesLen);
			}
		}
		else
		{
			pesLen -= writeLen;
			inLen -= writeLen;
			handleLen += writeLen;
		}
	}
}

int  CSMux::parseAAC(byte *inBuf,int inLen,byte **outBuf,int &outLen)
{
	*outBuf = new byte[inLen+5];
	outLen = 0;
	byte *&outbuf = *outBuf;
	//var outlen int = 0
	if (inBuf[1] == 0x00)
	{ //AACPacketTypeΪ0x00˵����AAC sequence header
		mAudioSpecificConfig.mObjectType = (inBuf[2] >> 3);
		mAudioSpecificConfig.mSamplerateIndex = (inBuf[2]&0x07)<<1 | inBuf[3]>>7;
		mAudioSpecificConfig.mChannels = (inBuf[3] << 1) >> 4;
		mAudioSpecificConfig.mFramLengthFlag = (inBuf[3] & 4) >> 2;
		mAudioSpecificConfig.mDependOnCCoder = (inBuf[3] & 2) >> 1;
		mAudioSpecificConfig.mExtensionFlag = inBuf[3] & 1;

		if (mAudioSpecificConfig.mExtensionFlag > 0) 
		{

		}
		//AAC(inBuf[2:], 2)
		mAACFLag = false;
		delete[] outbuf;
		outbuf = NULL;
		return 0;
	} 
	else 
	{
		int aacFrameLength = inLen + 5; //�����Ǵ���ADTSͷ7�ֽڵ�ȫ֡���ȣ�����֡���ݳ���
		outbuf[0] = 0xFF;
		outbuf[1] = 0xF1;
		outbuf[2] = ((mAudioSpecificConfig.mObjectType - 1) << 6) |
			((mAudioSpecificConfig.mSamplerateIndex&0x0F) << 2) |
			(mAudioSpecificConfig.mChannels >> 7);
		outbuf[3] = (mAudioSpecificConfig.mChannels << 6) | (byte)((aacFrameLength&0x1800)>>11);
		outbuf[4] = (byte)((aacFrameLength&0x7f8) >> 3);
		outbuf[5] = (byte)((aacFrameLength&0x7)<<5) | 0x1f;
		outbuf[6] = 0xFC | 0x00;
		/*
			//adts_fixed_header
				syncword 				//12 ͬ��ͷ ����0xFFF, all bits must be 1��������һ��ADTS֡�Ŀ�ʼ
				ID						//1 MPEG Version: 0 for MPEG-4, 1 for MPEG-2
				Layer					//2 always: '00'
				protectionAbsent		//1
				profile					//2 ��ʾʹ���ĸ������AAC����ЩоƬֻ֧��AAC LC ����MPEG-2 AAC�ж�����3�֣�
				samplingFrequencyIndex 	//4 ��ʾʹ�õĲ������±�
				privateBit				//1
				channelConfiguration	//3
				originalCopy			//1
				home					//1

			//adts_variable_header
				copyrightIBit			//1
				copyrightIStart			//1
				frameLength	 			//13 һ��ADTS֡�ĳ��Ȱ���ADTSͷ��AACԭʼ��.
				adtsBufferFullness 	    //11 0x7FF ˵�������ʿɱ������
				NumOfRawDataBlockInFrame//2*/

		memcpy(outbuf+7, inBuf+2,inLen-2);
		outLen = inLen + 5;
		return 1;
	}
	delete[] outbuf;
	outbuf = NULL;
	return 0;
}

int  CSMux::parseAACEx(byte *inBuf,int inLen,byte frameType,uint32 timestamp,byte pusi,byte afc,
				byte *mcc,int16 pid,TsChunkArray **tca,uint64 &pts64,int &pesLen)
{
	//var outlen int = 0
	if (inBuf[1] == 0x00)
	{ 
		assert(0);
	} 
	else 
	{
		int aacFrameLength = inLen + 5; //�����Ǵ���ADTSͷ7�ֽڵ�ȫ֡���ȣ�����֡���ݳ���
		byte outbuf[7] = {0};
		outbuf[0] = 0xFF;
		outbuf[1] = 0xF1;
		outbuf[2] = ((mAudioSpecificConfig.mObjectType - 1) << 6) |
			((mAudioSpecificConfig.mSamplerateIndex&0x0F) << 2) |
			(mAudioSpecificConfig.mChannels >> 7);
		outbuf[3] = (mAudioSpecificConfig.mChannels << 6) | (byte)((aacFrameLength&0x1800)>>11);
		outbuf[4] = (byte)((aacFrameLength&0x7f8) >> 3);
		outbuf[5] = (byte)((aacFrameLength&0x7)<<5) | 0x1f;
		outbuf[6] = 0xFC | 0x00;
		/*
			//adts_fixed_header
				syncword 				//12 ͬ��ͷ ����0xFFF, all bits must be 1��������һ��ADTS֡�Ŀ�ʼ
				ID						//1 MPEG Version: 0 for MPEG-4, 1 for MPEG-2
				Layer					//2 always: '00'
				protectionAbsent		//1
				profile					//2 ��ʾʹ���ĸ������AAC����ЩоƬֻ֧��AAC LC ����MPEG-2 AAC�ж�����3�֣�
				samplingFrequencyIndex 	//4 ��ʾʹ�õĲ������±�
				privateBit				//1
				channelConfiguration	//3
				originalCopy			//1
				home					//1

			//adts_variable_header
				copyrightIBit			//1
				copyrightIStart			//1
				frameLength	 			//13 һ��ADTS֡�ĳ��Ȱ���ADTSͷ��AACԭʼ��.
				adtsBufferFullness 	    //11 0x7FF ˵�������ʿɱ������
				NumOfRawDataBlockInFrame//2*/
		writeES((char *)outbuf,7,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		writeES((char *)inBuf+2,inLen-2,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
		return 1;
	}
	return 0;
}

int  CSMux::parseNAL(byte *inBuf,int inLen,byte **outBuf,int &outLen)
{
	*outBuf = new byte[inLen+256];
	byte *&outbuf  = *outBuf;
	int &outlen = outLen;
	int NALlen = 0;
	byte curNALtype = 0;

	//byte NALtype = 0;
	if (inBuf[1] == 0x00) 
	{ 
		//AVCPacketTypeΪ0x00˵����SPS/PPS
		int pos = 0;
		mSpsPpsNalLen = 0;
		/*byte configurationVersion = inBuf[5];
		byte AVCProfileIndication = inBuf[6];
		byte profile_compatibility = inBuf[7];
		byte AVCLevelIndication = inBuf[8];
		byte lengthSizeMinusOne = inBuf[9];*/                                  //<- �ǳ���Ҫ���� H.264 ��Ƶ�� NALU �ĳ��ȣ����㷽���� 1 + (lengthSizeMinusOne & 3)
		int numOfSequenceParameterSets = (int)(inBuf[10] & 0x1F);              //<- SPS �ĸ��������㷽���� numOfSequenceParameterSets & 0x1F
		int sequenceParameterSetLength = ((int)(inBuf[11]))<<8 | (int)(inBuf[12]); // <- SPS �ĳ���
		//fmt.Printf("numOfSequenceParameterSets sequenceParameterSetLength %d %d", numOfSequenceParameterSets, sequenceParameterSetLength)
		for (int i = 0; i < numOfSequenceParameterSets; i++)
		{
			mSpsPpsNal[mSpsPpsNalLen] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+1] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+2] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+3] = 0x01;
			memcpy(mSpsPpsNal+mSpsPpsNalLen+4, inBuf+(13+i*sequenceParameterSetLength),sequenceParameterSetLength/*(13+(i+1)*sequenceParameterSetLength)-(13+i*sequenceParameterSetLength)*/);
			mSpsPpsNalLen += (4 + sequenceParameterSetLength);
		}

		pos = 13 + sequenceParameterSetLength*numOfSequenceParameterSets;

		int numOfPictureParameterSets = (int)(inBuf[pos]);                          //<- PPS �ĸ���
		int pictureParameterSetLength = ((int)(inBuf[pos+1]))<<8 | (int)(inBuf[pos+2]); //<- PPS �ĳ���
		for (int i = 0; i < numOfPictureParameterSets; i++)
		{
			mSpsPpsNal[mSpsPpsNalLen] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+1] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+2] = 0x00;
			mSpsPpsNal[mSpsPpsNalLen+3] = 0x01;
			memcpy(mSpsPpsNal+mSpsPpsNalLen+4, inBuf+(pos+3+i*pictureParameterSetLength),pictureParameterSetLength/*(pos+3+(i+1)*pictureParameterSetLength)-(pos+3+i*pictureParameterSetLength)*/);
			mSpsPpsNalLen += (4 + pictureParameterSetLength);
		}
		delete[] outbuf;
		outbuf = NULL;
		//var NALlenSize int = 1 + int(lengthSizeMinusOne&3)
		return 0;
	} 
	else if (inBuf[1] == 0x01)
	{
		//AVCPacketTypeΪslice
		for (int i = 5; i < inLen-4; )
		{
			NALlen = ((int)(inBuf[i]))<<24 | ((int)(inBuf[i+1]))<<16 | ((int)(inBuf[i+2]))<<8 | (int)(inBuf[i+3]);
			if (NALlen < 0)
			{
				//logs->debug("[ParserNAL] NALlen=%d",NALlen);
				break;
			}
			curNALtype = inBuf[i+4] & 0x1f;
			if (curNALtype > 0 /*&& curNALtype != 6*/ && NALlen < inLen)
			{
				outbuf[outlen] = 0x00;
				outbuf[outlen+1] = 0x00;
				outbuf[outlen+2] = 0x00;
				outbuf[outlen+3] = 0x01;

				if (NALlen > 655356)
				{
					//logs->debug("\n[outlen NALlen  curNALtype %d %d %d]", outlen, NALlen, curNALtype);
					//logs->debug("[ParserNAL] inLen NALlen inBuf[i-5] inBuf[i-4] inBuf[i-3] inBuf[i-2] inBuf[i-1] inBuf[i] inBuf[i+1] %d %d 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", 
					//	inLen, NALlen, inBuf[i-5], inBuf[i-4], inBuf[i-3], inBuf[i-2], inBuf[i-1], inBuf[i], inBuf[i+1]);
				}
				memcpy(outbuf+outlen+4, inBuf+i+4,NALlen);
				outlen += (NALlen + 4);
				//fmt.Printf("[ParserNAL] inLen NALlen inBuf[i+2] inBuf[i+3] %d %d %d %d", inLen, NALlen, inBuf[i+2], inBuf[i+3])
			} 
			else 
			{
				logs->debug("[ParserNAL] curNALtype=%d", curNALtype);
			}
			i += (NALlen + 4);
		}
		//Test(outbuf[:], outlen)
		return 1;
	}
	delete[] outbuf;
	outbuf = NULL;
	return -1;
	/*
	 FLV�е�NAL��TS�е�NAl��һ����TS������ 0x00 0x00 0x00 0x01�ָ�
	*/
	/*
			 Nalu_type:
		        0x67 (0 11 00111) SPS  			�ǳ���Ҫ       	type = 7
		        0x68 (0 11 01000) PPS  			�ǳ���Ҫ       	type = 8
		        0x65 (0 11 00101) IDR֡ �ؼ�֡  �ǳ���Ҫ 		type = 5
		        0x41 (0 10 00001) P֡   		��Ҫ         	type = 1
		        0x01 (0 00 00001) B֡   		����Ҫ        	type = 1
		        0x06 (0 00 00110) SEI			����Ҫ        	type = 6
	*/

}

int  CSMux::parseNALEx(byte *inBuf,int inLen,byte frameType,uint32 timestamp,byte pusi,byte afc,
				byte *mcc,int16 pid,TsChunkArray **tca,uint64 &pts64,int &pesLen)
{
	int NALlen = 0;
	byte curNALtype = 0;

	//byte NALtype = 0;
	if (inBuf[1] == 0x00) 
	{ 
		assert(0);
	} 
	else if (inBuf[1] == 0x01)
	{
		char spsPpsNal[4] = {0};
		spsPpsNal[0] = 0x00;
		spsPpsNal[1] = 0x00;
		spsPpsNal[2] = 0x00;
		spsPpsNal[3] = 0x01;
		//AVCPacketTypeΪslice
		for (int i = 5; i < inLen-4; )
		{
			NALlen = ((int)(inBuf[i]))<<24 | ((int)(inBuf[i+1]))<<16 | ((int)(inBuf[i+2]))<<8 | (int)(inBuf[i+3]);
			if (NALlen < 0)
			{
				//logs->debug("[ParserNAL] NALlen=%d",NALlen);
				break;
			}
			curNALtype = inBuf[i+4] & 0x1f;
			if (curNALtype > 0 /*&& curNALtype != 6*/ && NALlen < inLen)
			{
				writeES(spsPpsNal,4,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
				writeES((char *)(inBuf+i+4),NALlen,frameType,timestamp,pusi,afc,mcc,pid,tca,pts64,pesLen);
				if (NALlen > 655356)
				{
					//logs->debug("\n[outlen NALlen  curNALtype %d %d %d]", outlen, NALlen, curNALtype);
					//logs->debug("[ParserNAL] inLen NALlen inBuf[i-5] inBuf[i-4] inBuf[i-3] inBuf[i-2] inBuf[i-1] inBuf[i] inBuf[i+1] %d %d 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", 
					//	inLen, NALlen, inBuf[i-5], inBuf[i-4], inBuf[i-3], inBuf[i-2], inBuf[i-1], inBuf[i], inBuf[i+1]);
				}
			} 
			else 
			{
				logs->debug("[ParserNAL] curNALtype=%d", curNALtype);
			}
			i += (NALlen + 4);
		}
		//Test(outbuf[:], outlen)
		return 1;
	}
	return -1;
	/*
	 FLV�е�NAL��TS�е�NAl��һ����TS������ 0x00 0x00 0x00 0x01�ָ�
	*/
	/*
			 Nalu_type:
		        0x67 (0 11 00111) SPS  			�ǳ���Ҫ       	type = 7
		        0x68 (0 11 01000) PPS  			�ǳ���Ҫ       	type = 8
		        0x65 (0 11 00101) IDR֡ �ؼ�֡  �ǳ���Ҫ 		type = 5
		        0x41 (0 10 00001) P֡   		��Ҫ         	type = 1
		        0x01 (0 00 00001) B֡   		����Ҫ        	type = 1
		        0x06 (0 00 00110) SEI			����Ҫ        	type = 6
	*/

}

int  CSMux::parseAACLen(byte *inBuf,int inLen)
{
	int outLen = 0;
	if (inBuf[1] == 0x00)
	{ 
		return 0;
	} 
	else 
	{
		outLen = inLen + 5;
		return outLen;
	}
	return -1;
}

int  CSMux::parseNALLen(byte *inBuf,int inLen)
{
	int outlen = 0;
	int NALlen = 0;
	byte curNALtype = 0;

	//byte NALtype = 0;
	if (inBuf[1] == 0x00) 
	{ 
		return 0;
	} 
	else if (inBuf[1] == 0x01)
	{
		//AVCPacketTypeΪslice
		for (int i = 5; i < inLen-4; )
		{
			NALlen = ((int)(inBuf[i]))<<24 | ((int)(inBuf[i+1]))<<16 | ((int)(inBuf[i+2]))<<8 | (int)(inBuf[i+3]);
			if (NALlen < 0)
			{
				//logs->debug("[ParserNAL] NALlen=%d",NALlen);
				break;
			}
			curNALtype = inBuf[i+4] & 0x1f;
			if (curNALtype > 0 /*&& curNALtype != 6*/ && NALlen < inLen)
			{
				if (NALlen > 655356)
				{
					//logs->debug("\n[outlen NALlen  curNALtype %d %d %d]", outlen, NALlen, curNALtype);
					//logs->debug("[ParserNAL] inLen NALlen inBuf[i-5] inBuf[i-4] inBuf[i-3] inBuf[i-2] inBuf[i-1] inBuf[i] inBuf[i+1] %d %d 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", 
					//	inLen, NALlen, inBuf[i-5], inBuf[i-4], inBuf[i-3], inBuf[i-2], inBuf[i-1], inBuf[i], inBuf[i+1]);
				}
				outlen += (NALlen + 4);
				//fmt.Printf("[ParserNAL] inLen NALlen inBuf[i+2] inBuf[i+3] %d %d %d %d", inLen, NALlen, inBuf[i+2], inBuf[i+3])
			} 
			else 
			{
				logs->debug("[ParserNAL] curNALtype=%d", curNALtype);
			}
			i += (NALlen + 4);
		}
		return outlen;
	}
	return -1;
	/*
	 FLV�е�NAL��TS�е�NAl��һ����TS������ 0x00 0x00 0x00 0x01�ָ�
	*/
	/*
			 Nalu_type:
		        0x67 (0 11 00111) SPS  			�ǳ���Ҫ       	type = 7
		        0x68 (0 11 01000) PPS  			�ǳ���Ҫ       	type = 8
		        0x65 (0 11 00101) IDR֡ �ؼ�֡  �ǳ���Ҫ 		type = 5
		        0x41 (0 10 00001) P֡   		��Ҫ         	type = 1
		        0x01 (0 00 00001) B֡   		����Ҫ        	type = 1
		        0x06 (0 00 00110) SEI			����Ҫ        	type = 6
	*/
}

int  CSMux::packPESLen(byte *inBuf,int inLen,byte frameType,uint32 timestamp)
{
	int headLen = 0;	//����ͷ������
	int packLen = 0;	//����������
	uint64 pts = 0;
	uint64 dts = 0;

	uint64 cts = ((uint64)inBuf[2])<<16 | ((uint64)inBuf[3])<<8 | ((uint64)inBuf[4]); //ƫ����cts
	if (frameType == 'A')
	{
		cts = 0;
	}
	dts = ((uint64)timestamp) * 90;	//flv�м�¼��ʱ�����DTS(TS�е�ʱ����FLV�е�1/90)
	pts = dts + cts*90;

	if (pts < dts)
	{
		pts = dts;
		logs->warn("*** [packPES] ERROR timestamp:  pts < dts ***");
	}
	//��ͷ����ʼ��ǰ׺��������ʶ��PES������Ϣ3���ֹ��ɡ�����ʼ��ǰ׺����23��������0����1����1�����ɵ�
	//3 0x000001 3�ֽڵ�ǰ׺
	//1	0xe0-0xef:v 0xc0-0xef:a
	//2 ����
// 	byte head[1024] = {0};
// 	head[0] = 0;
// 	head[1] = 0;
// 	head[2] = 1;
// 	head[3] = 0xE0; //pos= 3 ԭʼ�������ͺ���Ŀ��ȡֵ��1100��1111֮��
// 		//������ʶ��, ����������Ƶ����Ƶ����������
// 	head[4] = 0x00; //pos= 4 ��ʾ�Ӵ��ֽ�֮��PES����(��λ�ֽ�)��
// 	head[5] = 0x03;
	headLen += 6;

	byte PDflags = 0;
	//������ֵδȷ��
	if (frameType == 'I')
	{
		//I֡
		PDflags = 0x03;
	} 
	else if (frameType == 'P') 
	{
		//P֡
		PDflags = 0x03;
	} 
	else if (frameType == 'B') 
	{
		//B֡
		PDflags = 0x03;
	} 
	else if (frameType == 'A')
	{
		//��Ƶ
// 		head[3] = 0xC0;
		PDflags = 0x02;
	}
	//��ͷʶ���־��12������
	//head[6] = 0x80 //10000000
// 	byte fixBit = 0x02;				 //2 ����ֽ�
// 	byte PESscramblingControl = 0;   //2 PES��Ч���صļ���ģʽ��0��ʾ�����ܣ������ʾ�û��Զ��塣
// 	byte PESpriority = 0;            //1 PES���ݰ������ȼ�
// 	byte dataAlignmentIndicator = 0; //1 Ϊʱ�������˷���ͷ��֮��������������������ж���ķ��ʵ�Ԫ����
// 	byte copyright = 0;              //1 ��Ȩ����ʾ�а�Ȩ
// 	byte originalOrCopy = 0;         //1 pos= 6 1��ʾԭʼ���ݣ�0��ʾ����
// 	head[6] = fixBit<<6 | PESscramblingControl<<4 | PESpriority<<3 | dataAlignmentIndicator<<2 | copyright<<1 | originalOrCopy;
// 
	byte PTSDTSflags = PDflags;      //2 10��ʾ����PTS�ֶΣ�11��ʾ����PTS��DTS�ֶΣ�00��ʾ������pts_dts��DTS��
	byte ESCRflag = 0;               //1 1��ʾESCR��PES�ײ����֣�0��ʾ������
	byte ESrateFlag = 0;             //1 1��ʾPES���麬��ES_rate�ֶΡ�0��ʾ�����С�
	byte DSMtrickModeFlag = 0;       //1 1��ʾ��λ��trick_mode_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�ֻ��DSM��Ч��
	byte additionalCopyInfoFlag = 0; //1 1��ʾ��copy_info_flag�ֶΣ�0��ʾ�����ִ��ֶΡ�
// 	byte PESCRCflag = 0;             //1 1��ʾPES��������CRC�ֶΣ�0��ʾ�����ִ��ֶΡ�
	byte PESextensionFlag = 0;       //1 pos= 7 1��ʾ��չ�ֶ���PES��ͷ���ڣ�0��ʾ��չ�ֶβ�����
// 	/*if curStream.script.videodatarate > 0 {
// 		ESCRflag = 1
// 	}*/
// 	head[7] = PTSDTSflags<<6 | ESCRflag<<5 | ESrateFlag<<4 | DSMtrickModeFlag<<3 | additionalCopyInfoFlag<<2 | PESCRCflag<<1 | PESextensionFlag;
// 	head[8] = 0; // PESheaderDataLength = 0;         //��ʾ��ѡ�ֶκ�����ֶ���ռ���ֽ�����

	headLen += 3;

	if ((PTSDTSflags&0x02) > 0)
	{
		headLen += 5;
		/*pts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|1|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		        pts1     pts2                            pts3
		*/
	}
	if ((PTSDTSflags&0x01) > 0) 
	{
		headLen += 5;
		/*dts
		 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
		|0|0|0|1|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|1|
		         dts1    dts2                            dts3
		*/
	}

	if (ESCRflag > 0){

	}

	if (ESrateFlag > 0) {

	}

	if (ESrateFlag > 0) {

	}

	if (DSMtrickModeFlag > 0) {

	}

	if (additionalCopyInfoFlag > 0) {

	}

	if (PESextensionFlag > 0) {
		//������ʵ����һ��
	}
	if (frameType == 'I') 
	{ 
		//H264��Ƶ����
		int	outNalLen = parseNALLen(inBuf, inLen);
		if (outNalLen <= 0)
		{ 
			//ΪSPS/PPS��Tag��������д��PES��(����I֡ǰ��I֡һ����)
			return 0;
		}

		packLen = headLen + mSpsPpsNalLen + 6 + outNalLen;
	} 
	else if (frameType == 'A') 
	{
		//fmt.Printf("[PackPES] frameType == 'A'")
		if (inBuf[0]>>4 == 10) 
		{ 			
			//AAC
			int  outAACLen = parseAACLen(inBuf, inLen);
			if (outAACLen > 0)
			{
				packLen = headLen + outAACLen;
			}
		} 
		else 
		{
			//��AACĿǰ���򵥴���
			packLen = headLen + inLen - 1;
		}
	} else {
		int	 outNalLen =parseNALLen(inBuf, inLen);
		packLen = headLen + outNalLen + 6;
	}
	return packLen;
}
//���PSI������ֻ���PAT��PMT��
void CSMux::packPSI()
{
	int PATlen = 0;
	int PMTlen = 0;
	byte *tPAT = NULL;
	byte *tPMT = NULL;
	//PAT
	byte head[5] = {0};

	/*byte synByte = 0x47;               //ͬ���ֽ�, �̶�Ϊ0x47,��ʾ�������һ��TS����
	byte transportErrorIndicator = 0 ;   //��������ָʾ��
	byte payloadUnitStartIndicator = 1;  //��Ч���ص�Ԫ��ʼָʾ��, 1����Ϊ������ʼ��
	byte transport_priority = 0;         //��������, 1��ʾ�����ȼ�,������ƿ����õ��������ò���
	var PID int16 = PATpid;                 //PID
	byte transportScramblingControl = 0; //������ſ���(����)
	byte adaptionFieldControl = 1;       //����Ӧ���� 01������Ч���أ�10���������ֶΣ�11����
	byte continuityCounter = 0;          //���������� һ��4bit�ļ���������Χ0-15(patpmt����������)
	head[0] = synByte;
	head[1] = transportErrorIndicator<<7 | payloadUnitStartIndicator<<6 | transport_priority<<5 | (byte)((PID<<3)>>11);
	head[2] = (byte)(PID);
	head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;*/
	head[0] = 0x47;
	head[1] = 0x40;
	head[2] = 0x00;
	head[3] = 0x10 | mpatmcc;
	mpatmcc = (mpatmcc + 1) % 16;
	head[4] = 0; //������ͷ���и�0x00��pointer_field
	setPAT(PMTpid,&tPAT, PATlen);
	memcpy(mPAT, head,sizeof(head));
	memcpy(mPAT+sizeof(head), tPAT,PATlen);
	for (int pos = PATlen + 5; pos < 188; pos++)
	{
		mPAT[pos] = 0xFF;
	}

	//PMT
	/*synByte = 0x47;               //ͬ���ֽ�, �̶�Ϊ,��ʾ�������һ��TS����
	transportErrorIndicator = 0;    //��������ָʾ��
	payloadUnitStartIndicator = 1;  //��Ч���ص�Ԫ��ʼָʾ��, 1����Ϊ������ʼ��
	transport_priority = 0;         //��������, 1��ʾ�����ȼ�,������ƿ����õ��������ò���
	PID = PMTpid;                   //PID
	transportScramblingControl = 0; //������ſ���(����)
	adaptionFieldControl = 1;       //����Ӧ���� 01������Ч���أ�10���������ֶΣ�11����
	continuityCounter = 0;          //���������� һ��4bit�ļ���������Χ0-15(patpmt����������)
	head[0] = synByte;
	head[1] = transportErrorIndicator<<7 | payloadUnitStartIndicator<<6 | transport_priority<<5 | (byte)((PID<<3)>>11);
	head[2] = (byte)(PID);
	head[3] = transportScramblingControl<<6 | adaptionFieldControl<<4 | continuityCounter;*/
	head[0] = 0x47;
	head[1] = 0x50;
	head[2] = 0x00;
	head[3] = 0x10 | mpmtmcc;
	mpmtmcc = (mpmtmcc + 1) % 16;
	head[4] = 0; //������ͷ���и�0x00��pointer_field
	setPMT(Apid, Vpid, PCRpid,&tPMT, PMTlen);
	memcpy(mPMT, head,sizeof(head));
	memcpy(mPMT+sizeof(head), tPMT,PMTlen);
	for (int pos = PMTlen + 5; pos < 188; pos++)
	{
		mPMT[pos] = 0xFF;
	}
}

void CSMux::reset()
{
	mheadFlag = false;
	minfoTag = false;

	mheadBufDL = 0;
	mTagSize = -1;
	mTagSizeBufDL = 0;
	mTagBufDLen = 0;

	mamcc = 0;
	mvmcc = 0;
	mpatmcc = 0;
	mpmtmcc = 0;
}

void CSMux::setPcr(uint64 DTS,byte **outBuf,int &outLen)
{
	*outBuf = new byte[6];
	outLen = 0;
	byte *&PCR = *outBuf;
	uint64 pcr = DTS;
	uint64 pcrExt = pcr >> 33;
	PCR[0] = (byte)((pcr>>25) & 0xff);
	PCR[1] = (byte)((pcr>>17) & 0xff);
	PCR[2] = (byte)((pcr>>9) & 0xff);
	PCR[3] = (byte)((pcr>>1) & 0xff);

	byte PCRbase2 = (byte)(pcr & 0x01);
	byte PCRExt1 = (byte)((pcrExt>>8) & 0x01);
	byte PCRExt2 = (byte)(pcrExt & 0xff);

	PCR[4] = (PCRbase2<<7) | 0 | PCRExt1;
	PCR[5] = PCRExt2;
	outLen = 6;
}

//����������Tag���ҽ�������ٴ����TS
int  CSMux::tag2ts(byte *inBuf,int inLen,byte **outBuf,int &outLen,byte &outType,uint64 &outPts)
{
	byte *PESbuf = NULL;
	int PESLen = 0;

	outPts = 0;
	uint64 &pts64 = outPts;

	byte *TSbuf = NULL;
	int TSlen = -1;

	*outBuf = NULL;
	byte *&TsStream = *outBuf;
	int &StreamLen = outLen;

	outType = 'A';
	byte &frameType = outType;
	//curTag.head
	mcurTag.mhead.mtagType = inBuf[0];
	mcurTag.mhead.mdataSize = (int)(inBuf[1])<<16 | (int)(inBuf[2])<<8 | (int)(inBuf[3]);
	mcurTag.mhead.mtimeStamp = (uint32)(inBuf[4])<<16 | (uint32)(inBuf[5])<<8 | (uint32)(inBuf[6]) | (uint32)(inBuf[7])<<24; // |2|3|4|1|
	mcurTag.mhead.mstreamId = (int)(inBuf[8])<<16 | (int)(inBuf[9])<<8 | (int)(inBuf[10]);
	if (inBuf[0] == 0x09 && (inBuf[11]&0x0f) != 7)
	{
		logs->debug("inBuf[0]  inBuf[1] inBuf[2] inBuf[3] inBuf[4] inBuf[5] inBuf[6] inBuf[7] inBuf[8] inBuf[9]  inBuf[10] inBuf[11] 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", 
			inBuf[0], inBuf[1], inBuf[2], inBuf[3], inBuf[4], inBuf[5], inBuf[6], inBuf[7], inBuf[8], inBuf[9], inBuf[10], inBuf[11]);
	}

	if (mcurTag.mhead.mtagType == 0x08)
	{
		//��ƵTag
		mcurTag.mflag = 'a';
		mcurTag.maudio = dealATag(inBuf+11, 1);
		mcurTag.mhead.mdeviation = 0;
		frameType = 'A';
		packPES(inBuf+11, inLen-11, frameType, mcurTag.mhead.mtimeStamp,&PESbuf,PESLen,pts64);
		if (PESLen > 0)
		{
			packTS(PESbuf, PESLen, frameType, 0x01, 0x01, &mamcc, Apid, mcurTag.mhead.mtimeStamp,&TSbuf,TSlen);
			if (TSlen > 0)
			{
				memcpy(TsStream+StreamLen, TSbuf,TSlen);
				StreamLen += TSlen;
				//CallBack(TSbuf, TSlen) //����ص�����
				delete[] TSbuf;
			}
			delete[] PESbuf;			
		}
	} 
	else if (mcurTag.mhead.mtagType == 0x09 && true)
	{ //��ƵTag

		mcurTag.mflag = 'V';
		mcurTag.mvideo = dealVTag(inBuf+11, inLen-11);
		mcurTag.mhead.mdeviation = (int)(inBuf[13])<<16 | (int)(inBuf[14])<<8 | (int)(inBuf[15]); //ƫ����cts
		frameType = 'P';
		if (mcurTag.mflag == 'V' && mcurTag.mvideo.mframType == 1 && mcurTag.mvideo.mcodeId == 7)
		{
			//�ؼ�֡
			frameType = 'I';
			packPES(inBuf+11, inLen-11, frameType, mcurTag.mhead.mtimeStamp,&PESbuf, PESLen,pts64);
			if (PESLen > 0) 
			{
				packPSI();
				//CallBack(PAT[:], 188)
				//CallBack(PMT[:], 188)
				packTS(PESbuf, PESLen, frameType, 0x01, 0x01, &mvmcc, Vpid, mcurTag.mhead.mtimeStamp,&TSbuf,TSlen);
				if (TSlen > 0)
				{
					memcpy(TsStream+StreamLen, TSbuf,TSlen);
					StreamLen += TSlen;
					//CallBack(TSbuf, TSlen) //����ص�����
					delete[] TSbuf;
				}
				delete[] PESbuf;				
			}
		} 
		else 
		{
			packPES(inBuf+11, inLen-11, frameType, mcurTag.mhead.mtimeStamp, &PESbuf, PESLen,pts64);
			if (PESLen > 0)
			{
				packTS(PESbuf, PESLen, frameType, 0x01, 0x01, &mvmcc, Vpid, mcurTag.mhead.mtimeStamp,&TSbuf,TSlen);
				if (TSlen > 0)
				{
					memcpy(TsStream+StreamLen, TSbuf,TSlen);
					StreamLen += TSlen;
					//CallBack(TSbuf, TSlen) //����ص�����
					delete[] TSbuf;
				}
				delete[] PESbuf;				
			}			
		}

	} 
	else if (mcurTag.mhead.mtagType == 0x12)
	{ //scriptTag
		mscript = dealSTag(inBuf+11, mcurTag.mhead.mdataSize);
		//packPSI()
		byte *outSDT = NULL;
		int  outSDTLen = 0;
		setSDT(&outSDT,outSDTLen);
		if (outSDTLen > 0)
		{
			memcpy(TsStream+StreamLen, outSDT,outSDTLen);
			StreamLen += 188;
			//CallBack(SDT[:], 188)
			/*CallBack(PAT[:], 188)
			CallBack(PMT[:], 188)*/
			delete[] outSDT;
			minfoTag = true;
		}
	}
	pts64 = (uint64)(mcurTag.mhead.mtimeStamp)*90 + (uint64)(mcurTag.mhead.mdeviation)*90;
	return 1;
}

byte &CSMux::getAmcc()
{
	return mamcc;
}

byte &CSMux::getVmcc()
{
	return mvmcc;
}

byte &CSMux::getPatmcc()
{
	return mpatmcc;
}

byte &CSMux::getPmtmcc()
{
	return mpmtmcc;
}

byte *CSMux::getPat()
{
	return mPAT;
}

byte *CSMux::getPmt()
{
	return mPMT;
}
