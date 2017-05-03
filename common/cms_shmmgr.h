#ifndef __SHM_MGR_H__
#define __SHM_MGR_H__

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

class CShmMgr
{
	private:
		struct ShmKey
		{
			char 	keyFilename[128];		
			key_t	_key;
		};
	public:
		enum
		{
			SHM_TYPE_UPDATE = 1,			
			SHM_TYPE_MAX
		};
		
		CShmMgr();
		~CShmMgr();

		void *createShm(int keyIndex, unsigned int size);
		void *getShm(int keyIndex, unsigned int size);
		void detach(void *shmaddr);
		static	CShmMgr* instance();
	private:
		ShmKey	m_key[SHM_TYPE_MAX];

	private:
		
		void createShmKey( int entityNum,std::string key) ;
		key_t makeShmKey(char * buff, int reserved);
		static CShmMgr* minstance;
	};

#endif

