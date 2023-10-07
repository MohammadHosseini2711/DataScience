FROM python:3.9.18

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir pyparsing pydot \
  && pip install scipy \
  && pip install numpy \
  && pip install matplotlib \
  && pip install pandas 

WORKDIR /usr/src/app