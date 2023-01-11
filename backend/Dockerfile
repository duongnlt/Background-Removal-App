FROM nvidia/cuda:11.7.0-devel-ubuntu18.04

ENV PATH="/usr/local/cuda-11.7/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}"
RUN apt-get update && \
apt-get install --no-install-recommends -y build-essential gcc curl ca-certificates python3.8 python3-dev python3-pip libgl1-mesa-glx && \
apt-get clean && rm -rf /var/lib/apt/lists/* && \
python3 -m pip install --upgrade pip && pip3 install --upgrade setuptools wheel && \
pip3 install pycuda

WORKDIR /app

COPY . .


RUN pip install --upgrade -r requirements.txt

CMD ["python3", "main.py"]