FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y python3.7 && apt-get install -y python3-pip

COPY ./app /app
COPY ./model /model
COPY requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD [ "app/main.py" ]