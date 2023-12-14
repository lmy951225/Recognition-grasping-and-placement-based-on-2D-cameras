#!/bin/bash

# start agilebot controller algorithms
ContainerAgilebotController=$(docker ps |grep -c agilebotController)
if [ "${ContainerAgilebotController}" -ge 1 ];then  
   docker stop agilebotController
fi
cd /home/adt/RobotSystem/AgilebotController || exit
gnome-terminal -x bash -c "docker run -it --rm --name=agilebotController --network=host   --ipc=host -v $PWD:/home/adt -v /home/adt/data/iia/acq:/root/host_dir hub.micro-i.com.cn:9443/eastwind/agilebot:v0.08 python3 -m grpc_module.server_"

