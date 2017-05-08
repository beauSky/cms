#ifndef __CMS_COMMON_URL_H__
#define __CMS_COMMON_URL_H__
#include <string>

#define PROTOCOL_HTTP	"http"
#define PROTOCOL_HTTPS	"https"
#define PROTOCOL_RTMP	"rtmp"
#define PROTOCOL_WS		"ws"
#define PROTOCOL_WSS	"wss"

typedef struct _LinkUrl
{
	std::string protocol;
	std::string host;
	std::string addr;
	unsigned short port;
	bool isDefault;
	std::string app;
	std::string instanceName;
	std::string uri;
}LinkUrl;

bool parseUrl(std::string url,LinkUrl &linkUrl);
#endif
