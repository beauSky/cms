#ifndef __CMS_CHAR_INT_H__
#define __CMS_CHAR_INT_H__

int bigInt16(char *b);
int bigInt24(char *b);
int bigInt32(char *b);
long long bigInt64(char *b);

unsigned int bigUInt16(char *b);
unsigned int bigUInt24(char *b);
unsigned int bigUInt32(char *b);
unsigned long long bigUInt64(char *b);

void bigPutInt16(char *b,short v);
void bigPutInt24(char *b,int v);
void bigPutInt32(char *b,int v);
void bigPutInt64(char *b,long long v);

int littleInt16(char *b);
int littleInt24(char *b);
int littleInt32(char *b);
long long littleInt64(char *b);
void littlePutInt16(char *b,short v);
void littlePutInt24(char *b,int v);
void littlePutInt32(char *b,int v);
void littlePutInt64(char *b,long long v);

#endif