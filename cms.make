CXX = g++
LINK = $(CXX)

#debug config
#CXXFLAGS = -g -O2 -Wall -D_REENTRANT -D_POSIX_C_SOURCE=200112L -D_FILE_OFFSET_BITS=64 -D__CMS_APP_DEBUG__ -D_CMS_APP_USE_TIME_
#LIBS = -L./lib/ ./lib/libssl.a ./lib/libcrypto.a ./decode/libH264_5.a -lpthread -ldl -lrt -ls2n
CXXFLAGS = -g -O2 -Wall -D_REENTRANT -D_POSIX_C_SOURCE=200112L -D_FILE_OFFSET_BITS=64 -D_CMS_APP_USE_TIME_
LIBS = -L./lib/ ./lib/libssl.a ./lib/libcrypto.a ./decode/libH264_5.a -lpthread -ldl -lrt -ls2n

CMS_EXE = ./objs/cms

#directory name
CMS_INC = -I./
DIR_OBJS = objs
CMS_DIR = ./$(DIR_OBJS)/

DIR_TS = ts
DIR_STATIC=static
DIR_STRATEGY=strategy
DIR_TASK_MGR=taskmgr
DIR_FLV_POOL=flvPool
DIR_PROTOCOL = protocol
DIR_NET = net
DIR_KCP = kcp
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
DIR_CJSON=cJSON

#------------------------------- ALL -------------------------------
all:mkcmsdir exe

#---- ts ----
CMS_DIR_TS_OBJS 	=  	$(CMS_DIR)$(DIR_TS)/cms_ts.o \
							$(CMS_DIR)$(DIR_TS)/cms_hls_mgr.o \
							$(CMS_DIR)$(DIR_TS)/cms_ts_chunk.o
							
# objs
$(CMS_DIR)$(DIR_TS)/cms_ts.o: ./$(DIR_TS)/cms_ts.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_TS)/cms_ts.o ./$(DIR_TS)/cms_ts.cpp
$(CMS_DIR)$(DIR_TS)/cms_hls_mgr.o: ./$(DIR_TS)/cms_hls_mgr.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_TS)/cms_hls_mgr.o ./$(DIR_TS)/cms_hls_mgr.cpp
$(CMS_DIR)$(DIR_TS)/cms_ts_chunk.o: ./$(DIR_TS)/cms_ts_chunk.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_TS)/cms_ts_chunk.o ./$(DIR_TS)/cms_ts_chunk.cpp
			
#---- static ----
CMS_DIR_STATIC_OBJS 	=  	$(CMS_DIR)$(DIR_STATIC)/cms_static.o \
							$(CMS_DIR)$(DIR_STATIC)/cms_static_common.o 
# objs
$(CMS_DIR)$(DIR_STATIC)/cms_static.o: ./$(DIR_STATIC)/cms_static.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STATIC)/cms_static.o ./$(DIR_STATIC)/cms_static.cpp
$(CMS_DIR)$(DIR_STATIC)/cms_static_common.o: ./$(DIR_STATIC)/cms_static_common.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STATIC)/cms_static_common.o ./$(DIR_STATIC)/cms_static_common.cpp

			
#---- strategy ----
CMS_DIR_STRATEGY_OBJS 	=  	$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o \
								$(CMS_DIR)$(DIR_STRATEGY)/cms_jitter.o \
								$(CMS_DIR)$(DIR_STRATEGY)/cms_duration_timestamp.o \
								$(CMS_DIR)$(DIR_STRATEGY)/cms_first_play.o \
								$(CMS_DIR)$(DIR_STRATEGY)/cms_jump_last_x_seconds.o
# objs
$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o: ./$(DIR_STRATEGY)/cms_fast_bit_rate.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_fast_bit_rate.o ./$(DIR_STRATEGY)/cms_fast_bit_rate.cpp
$(CMS_DIR)$(DIR_STRATEGY)/cms_jitter.o: ./$(DIR_STRATEGY)/cms_jitter.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_jitter.o ./$(DIR_STRATEGY)/cms_jitter.cpp
$(CMS_DIR)$(DIR_STRATEGY)/cms_duration_timestamp.o: ./$(DIR_STRATEGY)/cms_duration_timestamp.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_duration_timestamp.o ./$(DIR_STRATEGY)/cms_duration_timestamp.cpp
$(CMS_DIR)$(DIR_STRATEGY)/cms_first_play.o: ./$(DIR_STRATEGY)/cms_first_play.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_first_play.o ./$(DIR_STRATEGY)/cms_first_play.cpp
$(CMS_DIR)$(DIR_STRATEGY)/cms_jump_last_x_seconds.o: ./$(DIR_STRATEGY)/cms_jump_last_x_seconds.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_STRATEGY)/cms_jump_last_x_seconds.o ./$(DIR_STRATEGY)/cms_jump_last_x_seconds.cpp

			
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
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_ssl.o \
														$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_pump.o
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
$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_pump.o: ./$(DIR_PROTOCOL)/cms_flv_pump.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_PROTOCOL)/cms_flv_pump.o ./$(DIR_PROTOCOL)/cms_flv_pump.cpp

#---- kcp ----
CMS_DIR_KCP_OBJS 	=  	$(CMS_DIR)$(DIR_KCP)/ikcp.o 
# objs
$(CMS_DIR)$(DIR_KCP)/ikcp.o: ./$(DIR_KCP)/ikcp.c
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_KCP)/ikcp.o ./$(DIR_KCP)/ikcp.c

			
#---- net ----
CMS_DIR_NET_OBJS 	=  	$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o \
									$(CMS_DIR)$(DIR_NET)/cms_udp_timer.o \
									$(CMS_DIR)$(DIR_NET)/cms_udp_conn.o \
									$(CMS_DIR)$(DIR_NET)/cms_net_mgr.o \
									$(CMS_DIR)$(DIR_NET)/cms_net_thread.o \
									$(CMS_DIR)$(DIR_NET)/cms_net_var.o 
# objs
$(CMS_DIR)$(DIR_NET)/cms_udp_conn.o: ./$(DIR_NET)/cms_udp_conn.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_udp_conn.o ./$(DIR_NET)/cms_udp_conn.cpp
$(CMS_DIR)$(DIR_NET)/cms_udp_timer.o: ./$(DIR_NET)/cms_udp_timer.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_udp_timer.o ./$(DIR_NET)/cms_udp_timer.cpp
$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o: ./$(DIR_NET)/cms_tcp_conn.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_tcp_conn.o ./$(DIR_NET)/cms_tcp_conn.cpp
$(CMS_DIR)$(DIR_NET)/cms_net_mgr.o: ./$(DIR_NET)/cms_net_mgr.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_net_mgr.o ./$(DIR_NET)/cms_net_mgr.cpp
$(CMS_DIR)$(DIR_NET)/cms_net_thread.o: ./$(DIR_NET)/cms_net_thread.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_net_thread.o ./$(DIR_NET)/cms_net_thread.cpp			
$(CMS_DIR)$(DIR_NET)/cms_net_var.o: ./$(DIR_NET)/cms_net_var.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_NET)/cms_net_var.o ./$(DIR_NET)/cms_net_var.cpp
			
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
														$(CMS_DIR)$(DIR_INTERFACE)/cms_stream_info.o \
														$(CMS_DIR)$(DIR_INTERFACE)/cms_conn_listener.o 
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
$(CMS_DIR)$(DIR_INTERFACE)/cms_stream_info.o: ./$(DIR_INTERFACE)/cms_stream_info.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_stream_info.o ./$(DIR_INTERFACE)/cms_stream_info.cpp
$(CMS_DIR)$(DIR_INTERFACE)/cms_conn_listener.o: ./$(DIR_INTERFACE)/cms_conn_listener.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_INTERFACE)/cms_conn_listener.o ./$(DIR_INTERFACE)/cms_conn_listener.cpp
			
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
													$(CMS_DIR)$(DIR_COMMON)/cms_utility.o \
													$(CMS_DIR)$(DIR_COMMON)/cms_time.o
											
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
$(CMS_DIR)$(DIR_COMMON)/cms_time.o: ./$(DIR_COMMON)/cms_time.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_COMMON)/cms_time.o ./$(DIR_COMMON)/cms_time.cpp

#----- app -----
CMS_DIR_APP_OBJS 	=  	$(CMS_DIR)$(DIR_APP)/cms_server.o \
											$(CMS_DIR)$(DIR_APP)/cms_app.o \
											$(CMS_DIR)$(DIR_APP)/cms_app_info.o
											
# objs
$(CMS_DIR)$(DIR_APP)/cms_server.o: ./$(DIR_APP)/cms_server.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_APP)/cms_server.o ./$(DIR_APP)/cms_server.cpp 
$(CMS_DIR)$(DIR_APP)/cms_app.o: ./$(DIR_APP)/cms_app.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_APP)/cms_app.o ./$(DIR_APP)/cms_app.cpp
$(CMS_DIR)$(DIR_APP)/cms_app_info.o: ./$(DIR_APP)/cms_app_info.cpp
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_APP)/cms_app_info.o ./$(DIR_APP)/cms_app_info.cpp
			
#----- cjson -----
CMS_DIR_CJSON_OBJS 	=  	$(CMS_DIR)$(DIR_CJSON)/cJSON.o
											
# objs
$(CMS_DIR)$(DIR_CJSON)/cJSON.o: ./$(DIR_CJSON)/cJSON.c
	$(CXX) -c $(CXXFLAGS) $(CMS_INC) -o \
			$(CMS_DIR)$(DIR_CJSON)/cJSON.o ./$(DIR_CJSON)/cJSON.c
			
#------------------------------- LINK -------------------------------
CMS_OBJS =	$(CMS_DIR_TS_OBJS) \
				$(CMS_DIR_STATIC_OBJS) \
				$(CMS_DIR_STRATEGY_OBJS) \
				$(CMS_DIR_TASK_MGR_OBJS) \
				$(CMS_DIR_FLV_POOL_OBJS) \
				$(CMS_DIR_PROTOCOL_OBJS) \
				$(CMS_DIR_KCP_OBJS) \
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
				$(CMS_DIR_APP_OBJS)    \
				$(CMS_DIR_CJSON_OBJS) 
				
exe: $(CMS_OBJS)
	@echo
	@echo ----------------------------- compile finish, then link -----------------------------
	@echo
	$(LINK) -o $(CMS_EXE) $(CMS_OBJS) $(LIBS)

#------------------------------- MKDIR -------------------------------
mkcmsdir:
	@test -d './$(DIR_OBJS)' || mkdir -p ./$(DIR_OBJS)
	@test -d './$(DIR_OBJS)/$(DIR_TS)' || mkdir -p ./$(DIR_OBJS)/$(DIR_TS)
	@test -d './$(DIR_OBJS)/$(DIR_STATIC)' || mkdir -p ./$(DIR_OBJS)/$(DIR_STATIC)
	@test -d './$(DIR_OBJS)/$(DIR_STRATEGY)' || mkdir -p ./$(DIR_OBJS)/$(DIR_STRATEGY)
	@test -d './$(DIR_OBJS)/$(DIR_TASK_MGR)' || mkdir -p ./$(DIR_OBJS)/$(DIR_TASK_MGR)
	@test -d './$(DIR_OBJS)/$(DIR_FLV_POOL)' || mkdir -p ./$(DIR_OBJS)/$(DIR_FLV_POOL)
	@test -d './$(DIR_OBJS)/$(DIR_PROTOCOL)' || mkdir -p ./$(DIR_OBJS)/$(DIR_PROTOCOL)
	@test -d './$(DIR_OBJS)/$(DIR_KCP)' || mkdir -p ./$(DIR_OBJS)/$(DIR_KCP)
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
	@test -d './$(DIR_OBJS)/$(DIR_CJSON)' || mkdir -p ./$(DIR_OBJS)/$(DIR_CJSON)

#------------------------------- CLEAN -------------------------------
clean:
	rm -rf ./$(DIR_OBJS)
















		
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			






























