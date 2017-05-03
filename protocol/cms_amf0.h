#ifndef __AMF_0_H__
#define __AMF_0_H__
#include <string>

/*********************************************************************************
  *Copyright(C),2010-2011,�Ʒ�����
  *FileName:	amf0.h
  *Author:		chengshuihua
  *Version:		1.0
  *Date:		2014.07.29
  *Description: ����amf0Э�����ݻ���amf0���ݿ飬Ŀǰ��֧�� Numeric,Boolean,String,
				Object,Array,MixedArray,Date�����ݵĽ���
  *Others:
  *History:
     1.Date:
       Author:
       Modification:
**********************************************************************************/

namespace amf0
{
	enum Amf0Type
	{
		AMF0_TYPE_NONE = -1,
		//�����Ǳ�׼����
		AMF0_TYPE_NUMERIC = 0x00,
		AMF0_TYPE_BOOLEAN,
		AMF0_TYPE_STRING,
		AMF0_TYPE_OBJECT,
		AMF0_TYPE_MOVIECLIP,
		AMF0_TYPE_NULL,
		AMF0_TYPE_UNDEFINED,
		AMF0_TYPE_REFERENCE,
		AMF0_TYPE_ECMA_ARRAY,
		AMF0_TYPE_OBJECT_END,
		AMF0_TYPE_STRICT_ARRAY,
		AMF0_TYPE_DATE,
		AMF0_TYPE_LONG_STRING,
		AMF0_TYPE_UNSUPPORTED,
		AMF0_TYPE_RECORD_SET,
		AMF0_TYPE_XML_OBJECT,
		AMF0_TYPE_TYPED_OBJECT,
		AMF0_TYPE_AMF3
	};
}

namespace amf0
{
	typedef unsigned char	 uint8;
	typedef short			 int16;
	typedef unsigned short	 uint16;
	typedef unsigned int	 uint32;
	typedef unsigned long long uint64;
	typedef struct __Amf0Node *pAmf0Node;
	struct __Amf0Data;

	/* string ���� */
	typedef struct __Amf0String {
		uint16 size;
		uint8 *mbstr;
	} Amf0String;

	/* array MixedArray ���� */
	//����ʾ MixedArray ʱ,first_element ��һ��Ԫ���� key,�ڶ����� value���������� key�����ĸ���vaule����������
	//����Ϊ size / 2
	typedef struct __Amf0Array {
		uint32 size;
		pAmf0Node first_element;
		pAmf0Node last_element;
	} Amf0Array;

	/* ����ڵ� */
	typedef struct __Amf0Node {
		__Amf0Data *data;
		pAmf0Node next;
		pAmf0Node prev;
	} Amf0Node;

	/* date ���� */
	typedef struct __Amf0Date {
		uint64 milliseconds;
		uint16 timezone;
	} Amf0Date;

	/* XML string ���� */
	typedef struct __Amf0XmlString {
		uint32 size;
		uint8 *mbstr;
	} Amf0XmlString;

	/* class ���� */
	typedef struct __Amf0Class {
		Amf0String name;
		Amf0Array elements;
	} Amf0Class;

	/* objects ���� */
	typedef struct __Amf0Data {
		uint8 type;
		union {
			double	number_data;
			uint8	boolean_data;
			Amf0String string_data;
			Amf0Array	array_data;
			Amf0Date	date_data;
			Amf0XmlString xmlstring_data;
			Amf0Class	class_data;
		};
	} Amf0Data;	

	/* amf0 ���ݿ� */
	typedef struct __Amf0Block
	{
		Amf0Array	array_data;
		std::string cmd; //����amf0���ݿ�ʱ�����ڻ�ȡ�������ͣ��������ݿ�ʱ������
	}Amf0Block;

	//�����ڴ�
	char		*bit64Reversal(char *bit);
	Amf0Data	*amf0DataNew(uint8 type);
	//��ȡ�ڴ��ֽ���
	uint32		amf0DataSize(Amf0Data *data);
	uint8		amf0DataGetType(Amf0Data *data);
	//Amf0Data	*amf0DataClone(Amf0Data *data);
	void		amf0DataFree(Amf0Data *data);

	/* ��amf0�ڵ����ɿ��ӻ��ַ��� */
	std::string amf0DataDumpString(Amf0Data *data,char *retract,bool key = false);
	/* ��amf0�ڵ�����amf0��ʽ�ַ��� */
	std::string amf0Data2String(Amf0Data *data,bool key = false);

	/* null ���� functions */
	Amf0Data	*amf0NullNew();
	Amf0Data    *amf0UndefinedNew();

	/* number ���� functions */
	Amf0Data	*amf0NumberNew(double value);
	double		amf0NumberGetValue(Amf0Data *data);
	void		amf0NumberSetValue(Amf0Data *data, double value);

	/* boolean ���� functions */
	Amf0Data	*amf0BooleanNew(uint8 value);
	uint8		amf0BooleanGetValue(Amf0Data *data);
	void		amf0BooleanSetValue(Amf0Data *data, uint8 value);

	/* string ���� functions */
	Amf0Data	*amf0StringNew(uint8 *str, uint16 size);
	Amf0Data	*amf0String(const char *str);
	uint16		amf0StringGetSize(Amf0Data *data);
	uint8		*amf0StringGetUint8Ts(Amf0Data *data);

	/* object ���� functions */
	Amf0Data    *amf0EcmaArrayNew();
	Amf0Data	*amf0ObjectNew(void);
	uint32		amf0ObjectSize(Amf0Data *data);
	Amf0Data	*amf0ObjectAdd(Amf0Data *data, const char *name, Amf0Data *element);
	Amf0Data	*amf0ObjectGet(Amf0Data *data, const char *name);
	Amf0Data	*amf0ObjectSet(Amf0Data *data, const char *name, Amf0Data *element);
	Amf0Data	*amf0ObjectDelete(Amf0Data *data, const char *name);
	Amf0Node	*amf0ObjectFirst(Amf0Data *data);
	Amf0Node	*amf0ObjectLast(Amf0Data *data);
	Amf0Node	*amf0ObjectNext(Amf0Node *node);
	Amf0Node	*amf0ObjectPrev(Amf0Node *node);
	Amf0Data	*amf0ObjectGetName(Amf0Node *node);
	Amf0Data	*amf0ObjectGetData(Amf0Node *node);

	/* array ���� functions */
	Amf0Data	*amf0ArrayNew(void);
	uint32		amf0ArraySize(Amf0Array *array);
	Amf0Data	*amf0ArrayPush(Amf0Array *array, Amf0Data *data);
	Amf0Data	*amf0ArrayPop(Amf0Array *array);
	Amf0Node	*amf0ArrayFirst(Amf0Array *array);
	Amf0Node	*amf0ArrayLast(Amf0Array *array);
	Amf0Node	*amf0ArrayNext(Amf0Node *node);
	Amf0Node	*amf0ArrayPrev(Amf0Node *node);
	Amf0Data	*amf0ArrayGet(Amf0Node *node);
	Amf0Data	*amf0ArrayGetAt(Amf0Array *array, uint32 n);
	Amf0Data	*amf0ArrayDelete(Amf0Array *array, Amf0Node *node);
	Amf0Data    *amf0ArrayDeleteAt(Amf0Array *array, uint32 n);
	Amf0Data	*amf0ArrayInsertBefore(Amf0Array *array, Amf0Node *node, Amf0Data *element);
	Amf0Data	*amf0ArrayInsertAfter(Amf0Array *array, Amf0Node *node, Amf0Data *element);

	/* date ���� functions */
	Amf0Data	*amf0DateNew(uint64 milliseconds, int16 timezone);
	uint64		amf0DateGetMilliseconds(Amf0Data *data);
	int16		amf0DateGetTimezone(Amf0Data *data);
	time_t		amf0Date2Time(Amf0Data *data);

	/* amf ���ݿ� functions */
	Amf0Block   *amf0BlockNew(void);
	void		amf0BlockRelease(Amf0Block *block);
	Amf0Data	*amf0BlockPush(Amf0Block *block, Amf0Data *data);
	uint32		amf0BlockSize(Amf0Block *block);

	//��amf0�������ɿ��ӻ��ַ���
	std::string amf0BlockDump(Amf0Block *block);
	//��amf0��������amf0��ʽ
	std::string amf0Block2String(Amf0Block *block);
	/*
		ͨ��key����value�����������amf���ݿ飬ֱ���ҵ�key�������ϡ�
		����ֵ��ʾvalue������(����ֵֻ����:Numeric��Boolean��String��Date,
		Date����minseconds-zone��ʽ����,Boolean����true��false��ʽ����
		�����������:Object��Array��MixedArray,���������±���)��
		ֵ����string����ʽ������strValue��
	*/
	Amf0Type amf0Block5Value(Amf0Block *block,const char *key,std::string &strValue);
	/*
		ͨ��λ��pos������ȡvalue��ֻ�����block��array����������еݹ������
		����ֵ��ʾvalue������(����ֵֻ����:Numeric��Boolean��String��Date,
		Date����minseconds-zone��ʽ����,Boolean����true��false��ʽ����
		�����������:Object��Array��MixedArray,strValue����Ϊ�գ���������ֵΪ AMF0_TYPE_NONE)��
		ֵ����string����ʽ������strValue��
	*/
	Amf0Type amf0Block5Value(Amf0Block *block,int pos,std::string &strValue);

	void	 amf0BlockRemoveNode(Amf0Block *block,int pos);
	Amf0Data *amf0BlockGetAmf0Data(Amf0Block *block,int pos);

	Amf0Block   *amf0Parse(char *buf,int len);

	Amf0Data *amf0DataClone(Amf0Data * data);
}
#endif