# Agilebot Manipulator Controller Docker

## Docker Build
1. Download refresh Agilebot.IR.A-1.*-py3-none-any.whl from [resource](https://w8212nxs8v.feishu.cn/wiki/wikcnGdjN9A7Zs4peZACwlbEsKe) 
2. paste Agilebot.IR.A-1.*-py3-none-any.whl under current dir along with [Dockerfile](./Dockerfile)
3. run command to build docker image
```
cd AgilebotController/docker
docker build -t hub.micro-i.com.cn:9443/eastwind/agilebot:v0.** .
```
4. push the docker image if TEST PASS
```
docker push hub.micro-i.com.cn:9443/eastwind/agilebot:v0.**
```
## Docker Deployment

### Docker pull  
```
docker pull hub.micro-i.com.cn:9443/eastwind/agilebot:v0.**
```
### General docker deployment 
```
cd AgilebotController
docker run -it --rm -v $PWD:/home/adt --name agilebotController -v /home/adt/host_dir:/root/host_dir hub.micro-i.com.cn:9443/eastwind/agilebot:v0.**  python3 -m grpc_module.server_
[You can change /home/adt/host_dir to anywhere you want]
```

### Package project to .so and Docker deployment with DQI compability
```
cd AgilebotController  
docker run -it --rm -v $PWD:/home/adt hub.micro-i.com.cn:9443/eastwind/agilebot:v0.**  
python3 setup.py build_ext 
```

1. copy whole AgilebotController .so project(AgibotController/build/AgitboController) under /home/adt/RobotSystem/AgilebotController
2. update Docker image version in [AgilebotController.sh](./AgilebotController.sh)
3. copy AgilebotController/DockerDeployment/WeiyiRobotController.desktop to Desktop
