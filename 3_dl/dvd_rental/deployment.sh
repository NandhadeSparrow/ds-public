#!/bin/bash

# repo_name = dvd_st
# install aws cli, docker

# create aws ecr repo
# aws_region = eu-north-1
# aws_repo_uri = 509399624514.dkr.ecr.eu-north-1.amazonaws.com/dvd-st

docker build -t repo_name .
docker run -p 127.0.0.1:8501 repo_name
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 509399624514.dkr.ecr.eu-north-1.amazonaws.com/dvd-st
docker tag repo_name:latest 509399624514.dkr.ecr.eu-north-1.amazonaws.com/dvd-st:latest
docker push 509399624514.dkr.ecr.eu-north-1.amazonaws.com/dvd-st

# create aws ecs cluster
# create aws service for ecs cluster
# create aws task for ecs cluster
# open public ip fron task details