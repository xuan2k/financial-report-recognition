docker run --name tabdet\
           --mount type="bind",source="/home/xuan/Project/OCR/code/baseline/control",target="/workspace/source"\
           --mount type="bind",source="/home/xuan/Project/OCR/",target="/workspace/warehouse"\
           --gpus all\
           --shm-size=8GB\
           -p 9999\
           -it table_det:1.0.0