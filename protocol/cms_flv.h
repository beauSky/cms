#ifndef __CMS_PROTOCOL_FLV_H__
#define __CMS_PROTOCOL_FLV_H__
#include <string>

#ifndef VideoTypeAVC
#define VideoTypeAVC 0x07
#endif

#define BitGet(Number,pos) ((Number)>>(pos)&1)
std::string getAudioType(unsigned char type);
std::string getVideoType(unsigned char type);
int         getAudioSampleRates(char *tag);
int			getAudioFrameRate(char *tag,int len);
int         getAudioFrameRate(int audioSampleRates);
#endif
