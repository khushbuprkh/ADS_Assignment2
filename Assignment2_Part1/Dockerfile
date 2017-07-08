FROM ubuntu:latest

RUN apt-get update -q && apt-get install -yqq \
    apt-utils \
    git \
    vim \
    nano \
    ssh \
    gcc \
    make \
    build-essential \
    libkrb5-dev \
    sudo 
	
RUN apt-get install -y python python-dev python-distribute python-pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install boto3
RUN pip install -U boto
RUN pip install bokeh
RUN pip install jupyter
RUN pip install seaborn
RUN pip install sklearn
RUN pip install scipy


ADD properties_2016.csv properties_2016.csv
ADD train_2016_v2.csv train_2016_v2.csv
ADD run.sh run.sh
ADD config.json config.json
ADD dataingestion.py dataingestion.py
ADD RawDataEDAA.ipynb RawDataEDAA.ipynb

ENTRYPOINT ["python" , "dataingestion.py"]