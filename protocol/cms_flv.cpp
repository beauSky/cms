#include <protocol/cms_flv.h>
#include <stdio.h>
using namespace std;

const char *g_AudioType[] = {"Linear PCM, platform endian",
	"ADPCM",
	"MP3",
	"Linear PCM, little endian",
	"Nellymoser 16-kHz mono",
	"Nellymoser 8-kHz mono",
	"Nellymoser",
	"G.711 A-law logarithmic PCM",
	"G.711 mu-law logarithmic PCM",
	"reserved",
	"AAC",
	"Speex",
	"0xC Unknow type",
	"0xD Unknow type ",
	"MP3 8-Khz",
	"Device-specific sound"};

const char *g_VideoType[] = {"0x00 Unknow type",
	"JPEG (currently unused)",
	"Sorenson H.263",
	"Screen video",
	"On2 VP6",
	"On2 VP6 with alpha channel",
	"Screen video version 2",
	"AVC"};

const int g_AvprivMpeg4audioSampleRates[16] = {
	96000, 88200, 64000, 48000, 44100, 32000,
	24000, 22050, 16000, 12000, 11025, 8000, 7350
};

std::string getAudioType(unsigned char type)
{
	if (type <= sizeof(g_AudioType))
	{
		return g_AudioType[type];
	}
	char szType[128];
	snprintf(szType,sizeof(szType),"%.02X Unknow type",type);
	return szType;
}

std::string getVideoType(unsigned char type)
{
	if (type <= sizeof(g_VideoType))
	{
		return g_VideoType[type];
	}
	char szType[128];
	snprintf(szType,sizeof(szType),"%.02X Unknow type",type);
	return szType;
}

int getAudioSampleRates(char *tag)
{
	unsigned char uc = 0;
	uc |= BitGet(tag[2],2) << 3;
	uc |= BitGet(tag[2],1) << 2;
	uc |= BitGet(tag[2],0) << 1;
	uc |= BitGet(tag[3],7);
	if (uc < sizeof(g_AvprivMpeg4audioSampleRates))
	{
		return g_AvprivMpeg4audioSampleRates[uc];
	}
	return 44100;
}

int getAudioFrameRate(char *tag,int len)
{
	int audioFrameRate = 1000 / ((1024 * 1000) / 48000);
	unsigned char audioType  = (unsigned char)((tag[0] >> 4) & 0x0F);
	if (audioType >= 0x0A )
	{
		if (tag[1] == 0x00 )
		{
			if (len >= 4 )
			{						
				int audioSampleRates = getAudioSampleRates(tag);						
				if (audioSampleRates > 0)
				{
					audioFrameRate = 1000 / ((1024 * 1000) / audioSampleRates);
				} 
				else 
				{
					audioFrameRate = 43;
				}
			}
		}
	}
	return audioFrameRate;
}

int  getAudioFrameRate(int audioSampleRates)
{
	int audioFrameRate = 1000 / ((1024 * 1000) / 48000);
	if (audioSampleRates > 0)
	{
		audioFrameRate = 1000 / ((1024 * 1000) / audioSampleRates);
	} 
	else 
	{
		audioFrameRate = 43;
	}
	return audioFrameRate;
}