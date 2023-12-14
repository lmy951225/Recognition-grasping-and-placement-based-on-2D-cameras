
docker run --rm --gpus all \
--shm-size=4g \
--ipc=host \
--net=host \
--rm \
--name tritionServer \
-v $(pwd)/models:/models \
hub.micro-i.com.cn:9443/dad/tritonserver:22.11-dev tritonserver --model-repository=/models

# bash -c """
# pip install pyro-ppl==1.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install open3d-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# tritonserver --model-repository=/models
# """

# https://pypi.douban.com/simple/