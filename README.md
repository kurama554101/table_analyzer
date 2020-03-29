# Table Analyzer

## Introduction

* Table Analyzer is the service to analyze table data.
* This service use [Autogluon](https://github.com/awslabs/autogluon).
* This service enable to send the csv data to Kaggle competition.

## Environment

* Docker
* Docker-Compose
* git

## Setup

### Set kaggle environment

If you want to submit kaggle competition, you set "kaggle.json" file into "kaggle_config" folder.

The way of getting "kaggle.json" is described in the following page.
https://github.com/Kaggle/kaggle-api#api-credentials

### Build Container

To create the containers, you execute the following command.

```
$ docker-compose up
```

If you rebuild the docker image, please execute the following command.

```
$ docker-compose up --build
```

### Use

you access "localhost:8601"
