FROM ubuntu
WORKDIR /src

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN apt-get -y install python3-sklearn
COPY iris.py ./iris.py
CMD ["python3", "iris.py"]