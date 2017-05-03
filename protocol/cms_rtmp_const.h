#ifndef __RTMP_CONST_H__
#define __RTMP_CONST_H__

enum RtmpConnStatus
{
	RtmpStatusError = -1,
	RtmpStatusShakeNone = 0,
	RtmpStatusShakeC0C1,
	RtmpStatusShakeS0,
	RtmpStatusShakeS1,
	RtmpStatusShakeC0,
	RtmpStatusShakeC1,
	RtmpStatusShakeSuccess,
	RtmpStatusConnect,
	RtmpStatusCreateStream,
	RtmpStatusPublish,
	RtmpStatusPlay1,
	RtmpStatusPlay2,
	RtmpStatusPause,
	RtmpStatusStop
};


enum RtmpType
{
	RtmpTypeNone,
	RtmpClient2Play,			//��Ϊ�ͻ���ȥplay��
	RtmpClient2Publish,			//��Ϊ�ͻ���ȥpublish��
	RtmpServerBPlay,			//��Ϊ�������Ӧplay����
	RtmpServerBPublish,			//��Ϊ�������Ӧpublish����
	RtmpServerBPlayOrPublish	//��Ϊ����˿�ʼ״̬����ȷ��
};

//ʱ���
#define TIMESTAMP_EXTENDED	0xFFFFFF

//cs id
#define CHUNK_STREAM_ID_PROTOCOL		0x02
#define CHUNK_STREAM_ID_COMMAND			0x03
#define CHUNK_STREAM_ID_USER_CONTROL	0x04
#define CHUNK_STREAM_ID_OVERSTREAM      0x05
#define CHUNK_STREAM_VIDEO_AUDIO        0x06
#define CHUNK_STREAM_ID_PLAY_PUBLISH    0x08


#define HEADER_FORMAT_FULL						0x00  //��ͷ��header type��ʱ��������ݴ�С���������͡���ID
#define HEADER_FORMAT_SAME_STREAM				0x01  //��ͷ��header type��ʱ��������ݴ�С����������
#define HEADER_FORMAT_SAME_LENGTH_AND_STREAM	0x02  //��ͷ��header type��ʱ���
#define HEADER_FORMAT_CONTINUATION				0x03  //��ͷ��header type

//MESSAGE_TYPE_PING ͨ��
#define USER_CONTROL_STREAM_BEGIN		0x00 //�������ͻ��˷��ͱ��¼�֪ͨ�Է�һ������ʼ�����ÿ�������ͨѶ����Ĭ������£�������ڳɹ��شӿͻ��˽�����������֮���ͱ��¼����¼�IDΪ0���¼������Ǳ�ʾ��ʼ�����õ�����ID��
#define USER_CONTROL_STREAM_EOF			0x01 //�������ͻ��˷��ͱ��¼�֪ͨ�ͻ��ˣ����ݻط���ɡ����û�з��ж��������Ͳ��ٷ������ݡ��ͻ��˶��������н��յ���Ϣ��4�ֽڵ��¼����ݱ�ʾ���طŽ���������ID��
#define USER_CONTROL_STREAM_DRY			0x02 //�������ͻ��˷��ͱ��¼�֪ͨ�ͻ��ˣ�����û�и�������ݡ�����������һ��������û��̽�⵽��������ݣ��Ϳ���֪ͨ�ͻ������ݽߡ�4�ֽڵ��¼����ݱ�ʾ�ݽ�����ID
#define USER_CONTROL_SET_BUFFER_LENGTH	0x03 //�ͻ��������˷��ͱ��¼�����֪�Է��Լ��洢һ���������ݵĻ���ĳ��ȣ����뵥λ����������˿�ʼ����һ������ʱ���ͱ��¼����¼����ݵ�ͷ�ĸ��ֽڱ�ʾ��ID����4���ֽڱ�ʾ���泤�ȣ����뵥λ����
#define USER_CONTROL_STREAM_LS_RECORDED 0x04 //����˷��ͱ��¼�֪ͨ�ͻ��ˣ�������һ��¼������4�ֽڵ��¼����ݱ�ʾ¼������ID��
#define USER_CONTROL_PING_REQUEST		0x06 //�����ͨ�����¼����Կͻ����Ƿ�ɴ�¼�������4���ֽڵ��¼��������������ñ�����ı���ʱ�䡣�ͻ����ڽ��յ�kMsgPingRequest֮�󷵻�kMsgPingResponse�¼���
#define USER_CONTROL_PING_RESPONSE		0x07 //�ͻ��������˷��ͱ���Ϣ��Ӧping�����¼������ǽ���kMsgPingRequest���� �� ʱ�䡣


//��Ϣ���� 
#define MESSAGE_TYPE_NONE                0x00
#define MESSAGE_TYPE_CHUNK_SIZE          0x01
#define MESSAGE_TYPE_ABORT               0x02
#define MESSAGE_TYPE_ACK                 0x03
#define MESSAGE_TYPE_USER_CONTROL        0x04
#define MESSAGE_TYPE_WINDOW_SIZE         0x05
#define MESSAGE_TYPE_BANDWIDTH           0x06
#define MESSAGE_TYPE_DEBUG				 0x07
#define MESSAGE_TYPE_AUDIO               0x08
#define MESSAGE_TYPE_VIDEO               0x09
#define MESSAGE_TYPE_FLEX                0x0F
#define MESSAGE_TYPE_AMF3_SHARED_OBJECT  0x10
#define MESSAGE_TYPE_AMF3                0x11
#define MESSAGE_TYPE_INVOKE              0x12
#define MESSAGE_TYPE_AMF0_SHARED_OBJECT  0x13
#define MESSAGE_TYPE_AMF0                0x14
#define MESSAGE_TYPE_STREAM_VIDEO_AUDIO  0x16 //����FMS3��������������������,�������������а���AudioData��VideoData
#define MESSAGE_TYPE_CUSTOME_RANGE		 0x17 //�Զ�������
#define MESSAGE_TYPE_FIRST_VIDEO_AUDIO	 0x18 //�Զ�������
/* MESSAGE_TYPE_STREAM_VIDEO_AUDIO �����������ݣ���Ҫ�ٴν���
��;		��С(Byte)		���ݺ���
StreamType	1				�������ࣨ0x08=��Ƶ��0x09=��Ƶ��
MediaSize	3				ý�����������С
TiMMER		3				����ʱ���,��λ����
Reserve		4				����,ֵΪ0
MediaData	MediaSize		ý�����ݣ���Ƶ����Ƶ
TagLen		4				֡�Ĵ�С��ֵΪý�����������С+��������(MediaSize+1+3+3+4) */

//AMF
#define AMF0	0x00
#define AMF3	0x03

#define DEFAULT_CHUNK_SIZE		128
#define DEFAULT_RTMP_CHUNK_SIZE	(8 * 1024)
#define DEFAULT_WINDOW_SIZE		2500000
#define DEFAULT_BANDWIDTH_SIZE	2500000

enum AudioSupport
{
	SUPPORT_SND_NONE = 0x0001, //ԭʼ��Ƶ���ݣ���ѹ��
	SUPPORT_SND_ADPCM = 0x0002, //ADPCM ѹ��
	SUPPORT_SND_MP3 = 0x0004, //mp3 ѹ��
	SUPPORT_SND_INTEL = 0x0008, //û��ʹ��
	SUPPORT_SND_UNUSED = 0x0010, //û��ʹ��
	SUPPORT_SND_NELLY8 = 0x0020, //NellyMoser 8KHZѹ��
	SUPPORT_SND_NELLY = 0x0040, //NellyMoseѹ����5��11��22��44KHZ��
	SUPPORT_SND_G711A = 0x0080, //G711A ��Ƶѹ����ֻ����flash media server��
	SUPPORT_SND_G711U = 0x0100, //G711U��Ƶѹ����ֻ����flash media server��
	SUPPORT_SND_NELLY16 = 0x0200, //NellyMoser 16KHZѹ��
	SUPPORT_SND_AAC = 0x0400, //AAC�����
	SUPPORT_SND_SPEEX = 0x0800, //Speex��Ƶ
	SUPPORT_SND_ALL = 0x0fff //����RTMP֧�ֵ���Ƶ��ʽ
};
enum VideoSupport
{
	SUPPORT_VID_UNUSED = 0x0001, //������ֵ
	SUPPORT_VID_JPEG = 0x0002, //������ֵ
	SUPPORT_VID_SORENSON = 0x0004, //Sorenson Flash video
	SUPPORT_VID_HOMEBREW = 0x0008, //V1 screen sharing
	SUPPORT_VID_VP6 = 0x0010, //On2 video (Flash 8+)
	SUPPORT_VID_VP6ALPHA = 0x0020, //On2 video with alpha channel
	SUPPORT_VID_HOMEBREWV = 0x0040, //Screen sharing version 2(Flash 8+)
	SUPPORT_VID_H264 = 0x0080, //H264 ��Ƶ
	SUPPORT_VID_ALL = 0x00ff //RTMP֧�ֵ�������Ƶ�������
};

enum BandWidthLimit
{
	BandWidth_Limit_Force = 0,	//�յ��������Ĵ��ڴ�С��Ϣ�����뷢�Ϳͻ��˴���ȷ��
	BandWidth_Limit_SOFT = 1,   //�ͻ��˿������÷��ͣ�
	BandWidth_Limit_Dynamic = 2 //����ȿ�����Ӳ����Ҳ������������
};


enum RtmpCommand
{
	RtmpCommandConnect,
	RtmpCommandCreateStream,
	RtmpCommandPlay,
	RtmpCommandReleaseStream,
	RtmpCommandPublish,
	RtmpCommandDeleteStream
};

#define		Amf0CommandConnect          "connect"
#define 	Amf0CommandCreateStream     "createStream"
#define 	Amf0CommandCloseStream      "closeStream"
#define 	Amf0CommandDeleteStream     "deleteStream"
#define 	Amf0CommandPlay             "play"
#define 	Amf0CommandPause            "pause"
#define 	Amf0CommandOnBwDone         "onBWDone"
#define 	Amf0CommandOnStatus         "onStatus"
#define 	Amf0CommandResult           "_result"
#define 	Amf0CommandError            "_error"
#define 	Amf0CommandReleaseStream    "releaseStream"
#define 	Amf0CommandFcPublish        "FCPublish"
#define 	Amf0CommandUnpublish        "FCUnpublish"
#define 	Amf0CommandPublish          "publish"
#define 	Amf0DataSampleAccess        "|RtmpSampleAccess"
#define 	Amf0SetDataFrame            "@setDataFrame"
#define 	Amf0MetaData                "onMetaData"
#define 	Amf0MetaCheckbw             "_checkbw"
#define 	Amf0SupportP2pReq           "supportPPReq"
#define 	Amf0SupportP2pRsp           "supportPPRsp"
#define 	Amf0PushBackServer          "pushBackServer"
#define 	Amf0CommandEdge             "edgeTT"       //�Զ�������
#define 	Amf0CommandYFPing           "yfping"       //�Զ�������
#define 	Amf0CommandYFResponse       "yfresponse"   //�Զ�������
#define 	Amf0CommandYFCheckDelay     "yfcheckdelay" //�Զ�������
#define 	Amf0CommandYFCheckDelayRsp  "yfdelayrsp"   //�Զ�������
	// FMLE
#define 	Amf0CommandOnFcPublish    "onFCPublish"
#define 	Amf0CommandOnFcUnpublish  "onFCUnpublish"
	// the signature for packets to client.
#define 	RtmpSigFmsVer    "3,5,3,888"
#define 	RtmpSigAmf0Ver   0
#define 	RtmpSigClientId  "quick rtmp"
	// onStatus consts.
#define 	StatusLevel        "level"
#define 	StatusCode         "code"
#define 	StatusDescription  "description"
#define 	StatusDetails      "details"
#define 	StatusClientId     "clientid"
#define 	StatusRedirect     "redirect"
	// status value
#define 	StatusLevelStatus  "status"
	// status error
#define 	StatusLevelError  "error"
	// code value
#define 	StatusCodeConnectSuccess        "NetConnection.Connect.Success"
#define 	StatusCodeConnectRejected       "NetConnection.Connect.Rejected"
#define 	StatusCodeStreamReset           "NetStream.Play.Reset"
#define 	StatusCodeStreamStart           "NetStream.Play.Start"
#define 	StatusCodeStreamPublishNotify   "NetStream.Play.PublishNotify"
#define 	StatusCodeStreamPause           "NetStream.Pause.Notify"
#define 	StatusCodeStreamUnpause         "NetStream.Unpause.Notify"
#define 	StatusCodePublishStart          "NetStream.Publish.Start"
#define 	StatusCodeStreamFailed          "NetStream.Play.Failed"
#define 	StatusCodeStreamStreamNotFound  "NetStream.Play.StreamNotFound"
#define 	StatusCodeDataStart             "NetStream.Data.Start"
#define 	StatusCodeUnpublishSuccess      "NetStream.Unpublish.Success"
	//description
#define 	DescriptionConnectionSucceeded  "Connection succeeded."
#define 	DescriptionConnectionRejected   "Connection rejected."

typedef struct _RtmpHeader
{
	unsigned int msgLength;
	unsigned int msgTypeID;
	unsigned int msgStreamID;
	unsigned int timestamp;
	unsigned int extendedTimestamp;
}RtmpHeader;

typedef struct 
{
	unsigned int	msgType;
	unsigned int	streamId;
	unsigned int	timestamp;
	unsigned int	absoluteTimestamp;
	unsigned int    bufLen;
	unsigned int	dataLen;
	char			*buffer;
}RtmpMessage;

typedef struct _OutBoundChunkStream
{
	unsigned int	id;
	RtmpHeader		*lastHeader;
	unsigned int	lastOutAbsoluteTimestamp;
	unsigned int	lastInAbsoluteTimestamp;
	unsigned int	startAtTimestamp;
}OutBoundChunkStream;

typedef struct _InboundChunkStream
{
	unsigned int	id;
	RtmpHeader		*lastHeader;
	unsigned int	lastOutAbsoluteTimestamp;
	unsigned int	lastInAbsoluteTimestamp;
	RtmpMessage		*currentMessage;
}InboundChunkStream;

#define  APP_NAME          "cms"
#define  NUM_VERSION       "1.0.0.0"
#define  APP_VERSION       "cms/1.0.0.0"
#define  SERVER_NAME       "franza"
#define  RTMP_VERSION      34013312
#endif