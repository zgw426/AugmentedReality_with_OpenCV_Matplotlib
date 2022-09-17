FROM python:3.10-buster

ARG work_dir="/work/"
# コンテナにアクセスした際のデフォルトディレクトリ
WORKDIR ${work_dir}

# pip更新
RUN pip install --upgrade pip

COPY requirements_*.txt ${work_dir}
RUN pip install -r requirements_fixed-version.txt

# MEASURES -> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
