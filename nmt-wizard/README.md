*Owner: Jean Senellart (jean.senellart (at) opennmt.net)*

# NWT-Wizard Hello Word

## Introduction

The goal of this tutorial is to configure a nmt-wizard server, and to launch a task for training on CPU a simple transliteration model from Russian to English, and to test the generated model.

Reference: [https://github.com/OpenNMT/nmt-wizard](https://github.com/OpenNMT/nmt-wizard)

## Server Configuration

- minimal environment requested: `python`, `pip`, `build-essential` , `make`

```
$ sudo apt-get update
$ sudo apt-get -y install python python-pip
```
- Set environment variable `TUTORIAL` with path to working directory for this tutorial, and change directory.

```
$ mkdir tutorial-onmt-wizard-1
$ export TUTORIAL=${PWD}/tutorial-onmt-wizard-1
$ cd ${TUTORIAL}
```

- Installation of redis

```
$ sudo apt-get -y install redis-server
```
or

```
$ curl http://download.redis.io/releases/redis-4.0.8.tar.gz > redis-4.0.8.tar.gz
$ tar xzf redis-4.0.8.tar.gz
$ cd redis-4.0.8
$ cd deps
$ make hiredis jemalloc linenoise lua geohash-int
$ cd ..
$ make
```

launch a server (chdir to src directory if you installed by compiling):

```
$ redis-server
```

And configure keyspace event handling in a new terminal:
```
$ redis-cli config set notify-keyspace-events Klgx
```

The Redis database contains the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `active` | list | Active tasks |
| `beat:<task_id>` | int | Specific ttl-key for a given task |
| `lock:<resource...,task:…>` | value | Temporary lock on a resource or task |
| `queued:<service>` | list | Tasks waiting for a resource |
| `resource:<service>:<resourceid>` | list | Tasks using this resource |
| `task:<taskid>` | dict | <ul><li>status: [queued, allocated, running, terminating, stopped]</li><li>job: json of jobid (if status>=waiting)</li><li>service:the name of the service</li><li>resource: the name of the resource - or auto before allocating one message: error message (if any), ‘completed’ if successfully finished</li><li>container_id: container in which the task run send back by docker notifier</li><li>(queued|allocated|running|updated|stopped)_time: time for each event</li></ul> |
| `files:<task_id>` | dict | files associated to a task, "log" is generated when training is complete |
| `queue:<task_id>` | str | expirable timestamp on the task - is used to regularily check status |
| `work` | list | Tasks to process |

- virtual env installation

```
$ pip install virtualenv
$ virtualenv ${TUTORIAL}
```

- get github project

```
$ cd ${TUTORIAL}
$ git clone https://github.com/OpenNMT/nmt-wizard.git wizard
```

- install docker

```
$ sudo apt-get install \
apt-transport-https \
ca-certificates \
curl \
software-properties-common
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce
$ sudo sudo usermod -aG docker {{YOURUSERNAME}}
```
or
for other OS please see [installation instructions here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

close the session and open a new one:
```
$ export TUTORIAL=${PWD}/tutorial-onmt-wizard-1
$ cd ${TUTORIAL}
$ docker run hello-world
```

- installation of python dependencies

```
$ cd nmt-wizard
$ sudo pip install -r requirements.txt
```

- create a public/private key, and add public key in `.ssh/authorized_keys` in order to enable connection from your server to your server without authentication (useful for remote servers).

```
$ ssh-keygen
```
command-line for local server
```
$ cat /home/{{YOURUSERNAME}}/.ssh/id_rsa.pub >> /home/{{YOURUSERNAME}}/.ssh/authorized_keys
```

## Data preparation

The data directory contains aligned space-tokenized russian-english names, training file and test file.

## Wizard Configuration

### Service configuration
The REST server and worker are configured by `nmt-wizard/server/settings.ini`. The LAUNCHER_MODE environment variable (defaulting to Production) can be set to select different set of options in development or production.
```
[DEFAULT]
# config_dir with service configuration
config_dir = ./config
# logging level
log_level = INFO
# refresh rate
refresh = 60

[Production]
redis_host = localhost
redis_port = 6379
redis_db   = 0
#redis_password=xxx
```
Here we use the default host and port of redis server.
You can choose different level of logging: `log_level`: `INFO`,`WARN`,`DEBUG`,`FATAL`,`ERROR`, for example, `DEBUG` gives the most complete log for debugging purpose.

### Local SSH server

We will define as the service for the tutorial, the local computer using `services.ssh` connector. Other connectors are for instance `service.ec2` or `service.torque`.

Get your IP with `ifconfig` - refered as `{{YOURIP}}` below.

Copy the following json file `nmt-wizard/server/config/default.json`.

```
{
    "docker": {
        "registries": {
            "dockerhub": {
                "type": "dockerhub",
                "uri": ""
            }
        }
    },
    "storages" : {
        "launcher": {
            "type": "http",
            "get_pattern": "<CALLBACK_URL>/file/<TASK_ID>/%s",
            "post_pattern": "<CALLBACK_URL>/file/<TASK_ID>/%s"
        }
    },
    "callback_url": "http://{{YOURIP}}:5000",
    "callback_interval": 60
}
```
Make sure to replace `{{YOURIP}}` by the actual IP.
* The first part, defines the registry named `dockerhub` as official public dockerhub registry. You could also define private docker hub registries, or use AWS Elastic Container Service (ECS) registries.
* the second part is defining the storage name `launcher` - as a simple http storage server implemented within the launcher. You will see below how to define other types of storage.

```
$ mkdir ${TUTORIAL}/inftraining_logs
```
Copy the following JSON into the `nmt-wizard/server/config/myserver.json`.
```
{
    "name": "myserver",
    "description": "My computing server",
    "module": "services.ssh",
    "variables": {
        "server_pool": [
            {
                "host": "localhost",
                "gpus": [0],
                "login": "{{YOURUSERNAME}}",
                "log_dir": "${TUTORIAL}/inftraining_logs"
            }
        ]
    },
    "privateKey": "/home/{{YOURUSERNAME}}/.ssh/id_rsa",
    "docker": {
        "mount": [
            "${TUTORIAL}/data/:/root/corpus/",
            "${TUTORIAL}/models:/root/models",
            "${TUTORIAL}/tmp:/root/tmp"
        ]
    }
}
```
This is a simple configuration of your server.
* `"gpus"` is set to off `[0]` since we're not using GPU in this tutorial
* the log file will be saved under `${TUTORIAL}/inftraining_logs`, make sure this directory exsit
* your SSH privateKey `/home/{{YOURUSERNAME}}/.ssh/id_rsa` will be used for connecting the server on which your publicKey  `/home/{{YOURUSERNAME}}/.ssh/id_rsa.pub` is authorized
* `${TUTORIAL}/corpus/` is your training corpus' directory on the local / remote server
* `${TUTORIAL}/models` is the directory for saving the models of `train` task
*  your custom files will be copied under `${TUTORIAL}/tmp`

## Launch the REST server

For production system, see the [Flask documentation](http://flask.pocoo.org/docs/0.12/deploying/) to deploy it for production.

In a terminal:
```
cd nmt-wizard/server
FLASK_APP=main.py flask run --host=0.0.0.0
```


## Launch the worker

In a new terminal:
```
$ export TUTORIAL=${PWD}/tutorial-onmt-wizard-1
$ cd ${TUTORIAL}
$ cd nmt-wizart/server
$ python worker.py
```

## The client commandline

{{YOURID}} is the trainer id, used as a prefix to generated models (default ENV[`LAUNCHER_TID`])
```
$ export TUTORIAL=${PWD}/tutorial-onmt-wizard-1
$ cd ${TUTORIAL}
$ export LAUNCHER_URL=http://{{YOURIP}}:5000
$ export LAUNCHER_TID={{YOURID}}
$ mkdir nmt-wizard/example
```

Copy the following JSON into the `nmt-wizard/example/helloworld.json`.

```
{
    "source": "ru",
    "target": "en",
    "data": {
        "sample_dist": [
            {
                "path": "train",
                "distribution": {
                    "helloworld.*": "1"
                }
            }
        ],
        "sample": 100000,
        "train_dir": "ru_en"
    },
    "options": {
        "train": {
            "rnn_size": "50",
            "word_vec_size": "20",
            "layers": "1",
            "src_vocab": "${TUTORIAL}/data/vocab/helloworld.ruen.src.dict",
            "tgt_vocab": "${TUTORIAL}/data/vocab/helloworld.ruen.tgt.dict"
        }
    }
}
```
This is a configuration of simple transliteration training task, it has two parts: `data` and `options`
* `data` part: the source language is `ru` and the target language is `en`, the corpus is picked from ${TUTORIAL}/corpus/`train_dir`/`path`/; the corpus which has extension `ru` \ `en` with pattern `helloworld.*` will be picked. Its coefficient is set to `1` in the total `10000` samples. see the [sampling documentation](https://github.com/OpenNMT/OpenNMT/blob/master/docs/training/sampling.md)
* `options` part: the configuration of training, in this training, a local custom file `${TUTORIAL}/vocab/helloworld.ruen.src.dict` will be copied and used on the server.  see the [training option documentation](https://github.com/OpenNMT/OpenNMT/blob/master/docs/options/train.md)


go through the different commands:

- `ls`： returns available services

```
python launcher.py ls
```
- `lt`： returns the list of tasks in the database

```
python launcher.py lt
```
- `launch` `train`： start training task, the return is a task id `taskid_1`

```
python launcher.py launch -s myserver -i nmtwizard/opennmt-lua -- -ms launcher: -c @../example/helloworld.json train
```
- `launch` `trans`： transliterate/translate `${TUTORIAL}/data/test/helloworld.ruen.test.ru` by using the model of `taskid_1`, the return is a task id `taskid_2`

```
python launcher.py launch -s myserver -i nmtwizard/opennmt-lua -- -ms launcher: -m <taskid_1> trans -i ${TUTORIAL}/data/test/helloworld.ruen.test.ru -o "launcher:helloworld.ruen.test.ru.out"
```
- `file`： get file from transaltion task

```
python launcher.py file -f helloworld.ruen.test.ru.out -k <taskid_2> > ${TUTORIAL}/data/test/helloworld.ruen.test.ru.out
```
- `terminate`：stop a running/queued task by its `taskid`

```
python launcher.py terminate -k <taskid>
```
- `status`：checks the status of a task by its `taskid`

```
python launcher.py status -k <taskid>
```
- `del`：delete a running/queued task by its `taskid`

```
python launcher.py del -k <taskid>
```

There are also other alternative storages
* S3: connecting to AWS server using access ID and secret key
* SSH: connecting to remote server hostname/IP via SSH

```
    "storages" : {
        "s3_models": {
            "type": "s3",
            "bucket": "model-catalog",
            "aws_credentials": {
                "access_key_id": "XXXXX",
                "secret_access_key": "XXXXX",
                "region_name": "eu-west-3"
            }
        },
        "myremoteserver": {
            "type": "ssh",
            "server": "myserver_url",
            "user": "XXXXX",
            "password": "XXXXX"
        }
```

---

*Congratulations for completing this Hello World tutorial!*
