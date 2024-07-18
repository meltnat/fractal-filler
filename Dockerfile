FROM python:3.12-bullseye

RUN apt-get update

ARG WORKDIR=/Workspace

RUN apt-get -y install locales && \
  localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV PYTHONPATH $WORKDIR

RUN mkdir $WORKDIR

WORKDIR $WORKDIR

RUN pip install --upgrade pip

COPY requirements.txt $WORKDIR

RUN pip install --no-cache-dir -r requirements.txt
