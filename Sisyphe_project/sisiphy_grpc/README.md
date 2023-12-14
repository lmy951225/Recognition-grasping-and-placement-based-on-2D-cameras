## Sisyphe client

Sisyphe client code. Before run client, first start AgilebotController and triton server. See details in Sisyphe_project/TritonModule.

### scripts
First start AgilebotController.

```
docker run -it --net host --ipc=host -w /home --name agilebotController -v $PWD:/{mount_path} hub.micro-i.com.cn:9443/eastwind/agilebot:sisiphy
python3 grpc_module/server_.py
```

Then start triton server. Finally start client by
```
python3 sisiphy_grpc/client_debug.py
```

