#!/bin/bash
if [ "$1" == "clean" ];then
	make -f cms.make clean
fi
version_file="./version.in"

if [ ! -f "$version_file" ]; then
	echo "1.0.0.0" > $version_file
fi

Major=`cat $version_file | awk -F '.' '{print $1}'`
Minor=`cat $version_file | awk -F '.' '{print $2}'`
Revision=`cat $version_file | awk -F '.' '{print $3}'`
Build=`cat $version_file | awk -F '.' '{print $4}'`
CmsLastVersion=$Major.$Minor.$Revision.$Build

echo ">>>>>>>>>>last version: $CmsLastVersion"

Build=$(($Build+1))

if [ $Build -gt 255 ];then
    Revision=$(($Revision+1))
	Build=0
fi

if [ $Revision -gt 255 ];then
    Minor=$(($Minor+1))
	Revision=0
fi

CmsVersion=$Major.$Minor.$Revision.$Build
echo ">>>>>>>>>>>cur version: $CmsVersion"

AppInfo="./app/cms_app_info.h"

echo "#ifndef __CMS_APP_INFO_H__" > $AppInfo
echo "#define __CMS_APP_INFO_H__" >> $AppInfo
echo "#include <string>" >> $AppInfo
echo "#define  APP_ALL_MODULE_THREAD_NUM 4" >> $AppInfo
echo "#define  APP_NAME \"cms(TK-Blue)\"" >> $AppInfo
echo "#define  APP_VERSION \"$CmsVersion\"" >> $AppInfo
echo "extern bool gcmsTestServer;" >> $AppInfo
echo "extern std::string gcmsTestUrl;" >> $AppInfo
echo "extern int  gcmsTestNum;" >> $AppInfo
echo "#endif" >> $AppInfo

make -f cms.make  && echo "$CmsVersion" > version.in



