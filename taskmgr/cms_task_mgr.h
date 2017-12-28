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
#ifndef __CMS_TASK_MGR_H__
#define __CMS_TASK_MGR_H__
#include <interface/cms_interf_conn.h>
#include <core/cms_lock.h>
#include <common/cms_type.h>
#include <core/cms_thread.h>
#include <string>
#include <map>
#include <queue>

#define CREATE_ACT_PULL	0x01
#define CREATE_ACT_PUSH	0x02

struct CreateTaskPacket 
{
	HASH				hash;
	std::string			pullUrl;
	std::string			pushUrl;
	std::string			oriUrl;
	std::string			refer;
	int					createAct;
	bool				isHotPush;
	bool				isPush2Cdn;      
	int64				ID;              //��������ID
};

class CTaskMgr
{
public:
	CTaskMgr();
	~CTaskMgr();
	static CTaskMgr *instance();
	static void freeInstance();

	static void *routinue(void *param);
	void thread();
	bool run();

	//��������ӿڻ��߱�����
	bool	pullTaskAdd(HASH &hash,Conn *conn);
	bool	pullTaskDel(HASH &hash);
	bool	pullTaskStop(HASH &hash);
	void	pullTaskStopAll();
	void    pullTaskStopAllByIP(std::string strIP);		//ɾ������ip
	bool	pullTaskIsExist(HASH &hash);
	//����������CDN����ӿ�
	bool	pushTaskAdd(HASH &hash,Conn *conn);
	bool	pushTaskDel(HASH &hash);
	bool	pushTaskStop(HASH &hash);
	void	pushTaskStopAll();
	void    pushTaskStopAllByIP(std::string strIP);		//ɾ������ip
	bool	pushTaskIsExist(HASH &hash);
	//�첽��������
	void	createTask(HASH &hash,std::string pullUrl,std::string pushUrl,std::string oriUrl,
		std::string refer,int createAct,bool isHotPush,bool isPush2Cdn);
	void	push(CreateTaskPacket *ctp);
private:
	bool	pop(CreateTaskPacket **ctp);
	//��������
	void	pullCreateTask(CreateTaskPacket *ctp);
	void	pushCreateTask(CreateTaskPacket *ctp);

	static CTaskMgr *minstance;
	bool			misRun;
	cms_thread_t	mtid;

	std::queue<CreateTaskPacket *>	mqueueCTP;
	CLock							mlockQueue;
	//��������
	CLock					mlockPullTaskConn;
	std::map<HASH,Conn *>	mmapPullTaskConn;
	//��������
	CLock					mlockPushTaskConn;
	std::map<HASH,Conn *>	mmapPushTaskConn;
};
#endif
