FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /

COPY . .

RUN pip install "paddleocr>=2.0.1" && \
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT [ "/bin/bash" ]