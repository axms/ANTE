FROM python:3.8
ENV TZ=America/Recife


WORKDIR /usr/src/ante

COPY requirements.txt ./

RUN apt-get update  && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential swig default-jre

RUN pip install --no-cache-dir -r requirements.txt  && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION v12.19.0
RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.1/install.sh | bash

RUN /bin/bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use --delete-prefix $NODE_VERSION"

ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH

RUN jupyter labextension install jupyterlab-plotly jupyterlab-dash

RUN useradd -ms /bin/bash ante

USER ante
