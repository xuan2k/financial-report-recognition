docker run --name tdcont\
           --mount type="bind",source="/home/xuan/Project/OCR/code/baseline/control",target="/workspace/source"\
           --mount type="bind",source="/home/xuan/Project/OCR/",target="/workspace/warehouse"\
           --gpus all\
           --shm-size=8GB\
           -p 8000:8000\
           -it pytorch/pytorch:1.4-cuda10.1-cudnn7-devel