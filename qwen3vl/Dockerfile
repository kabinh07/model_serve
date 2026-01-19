FROM pytorch/torchserve:latest-gpu

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y git

USER model-server

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install --upgrade bitsandbytes

COPY . .

CMD ["/bin/bash"]