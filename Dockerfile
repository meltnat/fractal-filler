FROM python:3.12-bullseye

RUN apt-get update

ARG USERNAME=pyuser
ARG GROUPNAME=pyuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG WORKDIR=/Workspace

RUN apt-get -y install locales && \
  localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV PYTHONPATH $WORKDIR

RUN groupadd --gid $USER_GID $GROUPNAME && \
  useradd -m --shell /bin/bash --uid $USER_UID --gid $USER_GID $USERNAME

RUN mkdir -p $WORKDIR && \
  chown -R $USER_UID:$USER_GID $WORKDIR

ENV PATH /home/$USERNAME/.local/bin:$PATH

USER $USERNAME

WORKDIR $WORKDIR

RUN pip install --upgrade --user pip

COPY requirements.txt $WORKDIR

RUN pip install --no-cache-dir --user -r requirements.txt