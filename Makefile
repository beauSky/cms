CXX = g++
CXXFLAGS = -g -O2 -Wall -D_REENTRANT -D_POSIX_C_SOURCE=200112L -D_FILE_OFFSET_BITS=64

LINK = $(CXX)
LIBS = -L./lib/ ./lib/libssl.a ./lib/libcrypto.a ./lib/libev.a -lpthread -ldl -lrt -ls2n

CMS_EXE = ./objs/cms

#directory name
CMS_INC = -I./
DIR_OBJS = objs
CMS_DIR = ./$(DIR_OBJS)/

DIR_JSON = json
DIR_STRATEGY=strategy
DIR_TASK_MGR=taskmgr
DIR_FLV_POOL=flvPool
DIR_PROTOCOL = protocol
DIR_NET = net
DIR_LOG = log
DIR_INTERFACE = interface
DIR_EV = ev
DIR_ENC = enc
DIR_DNSCACHE=dnscache
DIR_DISPATCH=dispatch
DIR_CORE=core
DIR_CONN=conn
DIR_CONFIG=config
DIR_COMMON=common
DIR_APP=app

#------------------------------- ALL -------------------------------
all:mkcmsdir exe

#---- json ----
CMS_DIR_JSON_OBJS 	=  	$(CMS_DIR)$(DIR_JSON)/json_reader.o \
														$(CMS_DIR)$(DIR_JSON)/json_value.o \
														$(CMS_DIR)$(DIR_JSON)/json_writer.o 
# objs
$(CMS_DIR)$(DIR_JSON)/json_reader.o: ./$(DIR_JSON)/json_reader.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_JSON)/json_reader.o ./$(DIR_JSON)/json_reader.cpp
$(CMS_DIR)$(DIR_JSON)/json_value.o: ./$(DIR_JSON)/json_value.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_JSON)/json_value.o ./$(DIR_JSON)/json_value.cpp
$(CMS_DIR)$(DIR_JSON)/json_writer.o: ./$(DIR_JSON)/json_writer.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_JSON)/json_writer.o ./$(DIR_JSON)/json_writer.cpp
			
#---- strategy ----
CMS_DIR_STRATEGY_OBJS 	=  	$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o 
# objs
$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o: ./$(DIR_STRATEGY)/cms_fast_bit_rate.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o ./$(DIR_STRATEGY)/cms_fast_bit_rate.cpp


#---- task mgr ----
CMS_DIR_TASK_MGR_OBJS 	=  	$(CMS_DIR)$(DIR_TASK_MGR)/cms_task_mgr.o 
# objs
$(CMS_DIR)$(DIR_TASK_MGR)/cms_task_mgr.o: ./$(DIR_TASK_MGR)/cms_task_mgr.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_TASK_MGR)/cms_task_mgr.o ./$(DIR_TASK_MGR)/cms_task_mgr.cpp

#---- flv pool ----
CMS_DIR_FLV_POOL_OBJS 	=  	$(CMS_DIR)$(DIR_FLV_POOL)/cms_flv_pool.o 
# objs
$(CMS_DIR)$(DIR_FLV_POOL)/cms_flv_pool.o: ./$(DIR_FLV_POOL)/cms_flv_pool.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_FLV_POOL)/cms_flv_pool.o ./$(DIR_FLV_POOL)/cms_flv_pool.cpp
			
#---- protocol ----
CMS_DIR_PROTOCOL_OBJS 	=  	$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp_handshake.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_amf0.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_transmission.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_http.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_ssl.o
# objs
$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp_handshake.o: ./$(DIR_PROTOCOL)/cms_rtmp_handshake.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp_handshake.o ./$(DIR_PROTOCOL)/cms_rtmp_handshake.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp.o: ./$(DIR_PROTOCOL)/cms_rtmp.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_rtmp.o ./$(DIR_PROTOCOL)/cms_rtmp.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv.o: ./$(DIR_PROTOCOL)/cms_flv.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv.o ./$(DIR_PROTOCOL)/cms_flv.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_amf0.o: ./$(DIR_PROTOCOL)/cms_amf0.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_amf0.o ./$(DIR_PROTOCOL)/cms_amf0.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_transmission.o: ./$(DIR_PROTOCOL)/cms_flv_transmission.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_transmission.o ./$(DIR_PROTOCOL)/cms_flv_transmission.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_http.o: ./$(DIR_PROTOCOL)/cms_http.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_http.o ./$(DIR_PROTOCOL)/cms_http.cpp
$(CMS_DIR)$(DIR_PROTOCOL)/cms_ssl.o: ./$(DIR_PROTOCOL)/cms_ssl.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_ssl.o ./$(DIR_PROTOCOL)/cms_ssl.cpp
#---- net ----
CMS_DIR_NET_OBJS 	=  	$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o
# objs
$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o: ./$(DIR_NET)/cms_tcp_conn.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o ./$(DIR_NET)/cms_tcp_conn.cpp

#---- log ----			
CMS_DIR_LOG_OBJS 	=  	$(CMS_DIR)$(DIR_LOG)/cms_log.o
# objs
$(CMS_DIR)$(DIR_LOG)/cms_log.o: ./$(DIR_LOG)/cms_log.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_LOG)/cms_log.o ./$(DIR_LOG)/cms_log.cpp

#---- interface ----
CMS_DIR_INTERFACE_OBJS 	=  	$(CMS_DIR)$(DIR_INTERFACE)/cms_thread.o \
														$(CMS_DIR)$(DIR_INTERFACE)/cms_dispatch.o \
														$(CMS_DIR)$(DIR_INTERFACE)/cms_interf_conn.o \
														$(CMS_DIR)$(DIR_INTERFACE)/cms_read_write.o \
														$(CMS_DIR)$(DIR_INTERFACE)/cms_protocol.o \
# objs
$(CMS_DIR)$(DIR_INTERFACE)/cms_thread.o: ./$(DIR_INTERFACE)/cms_thread.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_thread.o ./$(DIR_INTERFACE)/cms_thread.cpp
$(CMS_DIR)$(DIR_INTERFACE)/cms_dispatch.o: ./$(DIR_INTERFACE)/cms_dispatch.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_dispatch.o ./$(DIR_INTERFACE)/cms_dispatch.cpp
$(CMS_DIR)$(DIR_INTERFACE)/cms_interf_conn.o: ./$(DIR_INTERFACE)/cms_interf_conn.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_interf_conn.o ./$(DIR_INTERFACE)/cms_interf_conn.cpp
$(CMS_DIR)$(DIR_INTERFACE)/cms_read_write.o: ./$(DIR_INTERFACE)/cms_read_write.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_read_write.o ./$(DIR_INTERFACE)/cms_read_write.cpp
$(CMS_DIR)$(DIR_INTERFACE)/cms_protocol.o: ./$(DIR_INTERFACE)/cms_protocol.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_protocol.o ./$(DIR_INTERFACE)/cms_protocol.cpp
			
#----- ev -----
CMS_DIR_EV_OBJS 	=  	$(CMS_DIR)$(DIR_EV)/cms_ev.o
# objs
$(CMS_DIR)$(DIR_EV)/cms_ev.o: ./$(DIR_EV)/cms_ev.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_EV)/cms_ev.o ./$(DIR_EV)/cms_ev.cpp
			
#----- enc -----
CMS_DIR_ENC_OBJS 	=  	$(CMS_DIR)$(DIR_ENC)/cms_base64.o \
											$(CMS_DIR)$(DIR_ENC)/cms_crc32.o \
											$(CMS_DIR)$(DIR_ENC)/cms_crc32n.o \
											$(CMS_DIR)$(DIR_ENC)/cms_sha1.o
											
# objs
$(CMS_DIR)$(DIR_ENC)/cms_base64.o: ./$(DIR_ENC)/cms_base64.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_ENC)/cms_base64.o ./$(DIR_ENC)/cms_base64.cpp
$(CMS_DIR)$(DIR_ENC)/cms_crc32.o: ./$(DIR_ENC)/cms_crc32.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_ENC)/cms_crc32.o ./$(DIR_ENC)/cms_crc32.cpp
$(CMS_DIR)$(DIR_ENC)/cms_crc32n.o: ./$(DIR_ENC)/cms_crc32n.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_ENC)/cms_crc32n.o ./$(DIR_ENC)/cms_crc32n.cpp
$(CMS_DIR)$(DIR_ENC)/cms_sha1.o: ./$(DIR_ENC)/cms_sha1.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_ENC)/cms_sha1.o ./$(DIR_ENC)/cms_sha1.cpp

#----- dnscache -----
CMS_DIR_DNSCACHE_OBJS 	=  	$(CMS_DIR)$(DIR_DNSCACHE)/cms_dns_cache.o 
											
# objs
$(CMS_DIR)$(DIR_DNSCACHE)/cms_dns_cache.o: ./$(DIR_DNSCACHE)/cms_dns_cache.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_DNSCACHE)/cms_dns_cache.o ./$(DIR_DNSCACHE)/cms_dns_cache.cpp
			
#----- dispatch -----
CMS_DIR_DISPATCH_OBJS 	=  	$(CMS_DIR)$(DIR_DISPATCH)/cms_net_dispatch.o 
											
# objs
$(CMS_DIR)$(DIR_DISPATCH)/cms_net_dispatch.o: ./$(DIR_DISPATCH)/cms_net_dispatch.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_DISPATCH)/cms_net_dispatch.o ./$(DIR_DISPATCH)/cms_net_dispatch.cpp			
			
#----- core -----
CMS_DIR_CORE_OBJS 	=  	$(CMS_DIR)$(DIR_CORE)/cms_buffer.o \
												$(CMS_DIR)$(DIR_CORE)/cms_errno.o \
												$(CMS_DIR)$(DIR_CORE)/cms_lock.o \
												$(CMS_DIR)$(DIR_CORE)/cms_thread.o
											
# objs
$(CMS_DIR)$(DIR_CORE)/cms_buffer.o: ./$(DIR_CORE)/cms_buffer.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CORE)/cms_buffer.o ./$(DIR_CORE)/cms_buffer.cpp			
$(CMS_DIR)$(DIR_CORE)/cms_errno.o: ./$(DIR_CORE)/cms_errno.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CORE)/cms_errno.o ./$(DIR_CORE)/cms_errno.cpp			
$(CMS_DIR)$(DIR_CORE)/cms_lock.o: ./$(DIR_CORE)/cms_lock.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CORE)/cms_lock.o ./$(DIR_CORE)/cms_lock.cpp			
$(CMS_DIR)$(DIR_CORE)/cms_thread.o: ./$(DIR_CORE)/cms_thread.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CORE)/cms_thread.o ./$(DIR_CORE)/cms_thread.cpp	

#----- conn -----
CMS_DIR_CONN_OBJS 	=  	$(CMS_DIR)$(DIR_CONN)/cms_conn_rtmp.o \
												$(CMS_DIR)$(DIR_CONN)/cms_conn_mgr.o \
												$(CMS_DIR)$(DIR_CONN)/cms_http_s.o \
												$(CMS_DIR)$(DIR_CONN)/cms_http_c.o 
											
# objs
$(CMS_DIR)$(DIR_CONN)/cms_conn_rtmp.o: ./$(DIR_CONN)/cms_conn_rtmp.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CONN)/cms_conn_rtmp.o ./$(DIR_CONN)/cms_conn_rtmp.cpp
$(CMS_DIR)$(DIR_CONN)/cms_conn_mgr.o: ./$(DIR_CONN)/cms_conn_mgr.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CONN)/cms_conn_mgr.o ./$(DIR_CONN)/cms_conn_mgr.cpp
$(CMS_DIR)$(DIR_CONN)/cms_http_s.o: ./$(DIR_CONN)/cms_http_s.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CONN)/cms_http_s.o ./$(DIR_CONN)/cms_http_s.cpp
$(CMS_DIR)$(DIR_CONN)/cms_http_c.o: ./$(DIR_CONN)/cms_http_c.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CONN)/cms_http_c.o ./$(DIR_CONN)/cms_http_c.cpp

#----- config -----
CMS_DIR_CONFIG_OBJS 	=  	$(CMS_DIR)$(DIR_CONFIG)/cms_config.o
											
# objs
$(CMS_DIR)$(DIR_CONFIG)/cms_config.o: ./$(DIR_CONFIG)/cms_config.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CONFIG)/cms_config.o ./$(DIR_CONFIG)/cms_config.cpp 

#----- common -----
CMS_DIR_COMMON_OBJS 	=  	$(CMS_DIR)$(DIR_COMMON)/cms_binary_reader.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_binary_writer.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_char_int.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_shmmgr.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_url.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_utility.o
											
# objs
$(CMS_DIR)$(DIR_COMMON)/cms_binary_reader.o: ./$(DIR_COMMON)/cms_binary_reader.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_binary_reader.o ./$(DIR_COMMON)/cms_binary_reader.cpp
$(CMS_DIR)$(DIR_COMMON)/cms_binary_writer.o: ./$(DIR_COMMON)/cms_binary_writer.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_binary_writer.o ./$(DIR_COMMON)/cms_binary_writer.cpp
$(CMS_DIR)$(DIR_COMMON)/cms_char_int.o: ./$(DIR_COMMON)/cms_char_int.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_char_int.o ./$(DIR_COMMON)/cms_char_int.cpp
$(CMS_DIR)$(DIR_COMMON)/cms_shmmgr.o: ./$(DIR_COMMON)/cms_shmmgr.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_shmmgr.o ./$(DIR_COMMON)/cms_shmmgr.cpp
$(CMS_DIR)$(DIR_COMMON)/cms_url.o: ./$(DIR_COMMON)/cms_url.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_url.o ./$(DIR_COMMON)/cms_url.cpp
$(CMS_DIR)$(DIR_COMMON)/cms_utility.o: ./$(DIR_COMMON)/cms_utility.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_utility.o ./$(DIR_COMMON)/cms_utility.cpp

#----- app -----
CMS_DIR_APP_OBJS 	=  	$(CMS_DIR)$(DIR_APP)/cms_server.o \
											$(CMS_DIR)$(DIR_APP)/cms_app.o
											
# objs
$(CMS_DIR)$(DIR_APP)/cms_server.o: ./$(DIR_APP)/cms_server.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_APP)/cms_server.o ./$(DIR_APP)/cms_server.cpp 
$(CMS_DIR)$(DIR_APP)/cms_app.o: ./$(DIR_APP)/cms_app.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_APP)/cms_app.o ./$(DIR_APP)/cms_app.cpp
			
#------------------------------- LINK -------------------------------
CMS_OBJS =	$(CMS_DIR_STRATEGY_OBJS) \
				$(CMS_DIR_TASK_MGR_OBJS) \
				$(CMS_DIR_FLV_POOL_OBJS) \
				$(CMS_DIR_JSON_OBJS) \
				$(CMS_DIR_PROTOCOL_OBJS) \
				$(CMS_DIR_NET_OBJS) \
				$(CMS_DIR_LOG_OBJS) \
				$(CMS_DIR_INTERFACE_OBJS) \
				$(CMS_DIR_EV_OBJS) \
				$(CMS_DIR_ENC_OBJS) \
				$(CMS_DIR_DNSCACHE_OBJS) \
				$(CMS_DIR_DISPATCH_OBJS) \
				$(CMS_DIR_CORE_OBJS) \
				$(CMS_DIR_CONN_OBJS) \
				$(CMS_DIR_CONFIG_OBJS) \
				$(CMS_DIR_COMMON_OBJS) \
				$(CMS_DIR_APP_OBJS) 
				
exe: $(CMS_OBJS)
	@echo
	@echo ----------------------------- compile finish, then link -----------------------------
	@echo
	$(LINK) -o $(CMS_EXE) $(CMS_OBJS) $(LIBS)

#------------------------------- MKDIR -------------------------------
mkcmsdir:
	@test -d './$(DIR_OBJS)' || mkdir -p ./$(DIR_OBJS)
	@test -d './$(DIR_OBJS)/$(DIR_STRATEGY)' || mkdir -p ./$(DIR_OBJS)/$(DIR_STRATEGY)
	@test -d './$(DIR_OBJS)/$(DIR_TASK_MGR)' || mkdir -p ./$(DIR_OBJS)/$(DIR_TASK_MGR)
	@test -d './$(DIR_OBJS)/$(DIR_FLV_POOL)' || mkdir -p ./$(DIR_OBJS)/$(DIR_FLV_POOL)
	@test -d './$(DIR_OBJS)/$(DIR_JSON)' || mkdir -p ./$(DIR_OBJS)/$(DIR_JSON)
	@test -d './$(DIR_OBJS)/$(DIR_PROTOCOL)' || mkdir -p ./$(DIR_OBJS)/$(DIR_PROTOCOL)
	@test -d './$(DIR_OBJS)/$(DIR_NET)' || mkdir -p ./$(DIR_OBJS)/$(DIR_NET)
	@test -d './$(DIR_OBJS)/$(DIR_LOG)' || mkdir -p ./$(DIR_OBJS)/$(DIR_LOG)
	@test -d './$(DIR_OBJS)/$(DIR_INTERFACE)' || mkdir -p ./$(DIR_OBJS)/$(DIR_INTERFACE)
	@test -d './$(DIR_OBJS)/$(DIR_EV)' || mkdir -p ./$(DIR_OBJS)/$(DIR_EV)
	@test -d './$(DIR_OBJS)/$(DIR_ENC)' || mkdir -p ./$(DIR_OBJS)/$(DIR_ENC)
	@test -d './$(DIR_OBJS)/$(DIR_DNSCACHE)' || mkdir -p ./$(DIR_OBJS)/$(DIR_DNSCACHE)
	@test -d './$(DIR_OBJS)/$(DIR_DISPATCH)' || mkdir -p ./$(DIR_OBJS)/$(DIR_DISPATCH)	
	@test -d './$(DIR_OBJS)/$(DIR_CORE)' || mkdir -p ./$(DIR_OBJS)/$(DIR_CORE)
	@test -d './$(DIR_OBJS)/$(DIR_CONN)' || mkdir -p ./$(DIR_OBJS)/$(DIR_CONN)	
	@test -d './$(DIR_OBJS)/$(DIR_CONFIG)' || mkdir -p ./$(DIR_OBJS)/$(DIR_CONFIG)
	@test -d './$(DIR_OBJS)/$(DIR_COMMON)' || mkdir -p ./$(DIR_OBJS)/$(DIR_COMMON)
	@test -d './$(DIR_OBJS)/$(DIR_APP)' || mkdir -p ./$(DIR_OBJS)/$(DIR_APP)

#------------------------------- CLEAN -------------------------------
clean:
	rm -rf ./$(DIR_OBJS)
















		
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			






























