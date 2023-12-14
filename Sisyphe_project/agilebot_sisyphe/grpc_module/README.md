
python3 -m grpc_module.server_
python3 -m grpc_module.client_
sudo docker run -it --net host --ipc=host -w /home -v $PWD:/home hub.micro-i.com.cn:9443/eastwind/agilebot:v1 /bin/bash
sudo docker exec -it 14430ce6d5e2  /bin/bash
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. Agile_robot.proto

python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. --pyi_out=. agile_robot.proto


###note
sudo docker stop agilebotController
sudo docker rm agilebotController
sudo docker run -it --net host --ipc=host -w /home --name agilebotController -v $PWD:/home hub.micro-i.com.cn:9443/eastwind/agilebot:v0.08
python3 -m grpc_module.server_

sudo docker exec -it agilebotController /bin/bash
python3 -m grpc_module.client_grasp



source venv/bin/activate

rm -f Agile_robot_pb2.py
rm -f Agile_robot_pb2_grpc.py

