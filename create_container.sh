docker run --name tabel_tsr\
           --mount type="bind",source="/home/xuan/Project/OCR/code/baseline/control",target="/workspace/source"\
           --mount type="bind",source="/home/xuan/Project/OCR/sample",target="/workspace/warehouse"\
           --gpus all\
           --shm-size=8GB\
           -p 9999\
           -it table_tsr:1.0.3