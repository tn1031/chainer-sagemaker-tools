# chainer-sagemaker-tools

This repository is a collection of tools to run SageMaker jobs.

It contains

- CLIs : Some command line tools to use SageMaker easily. See below guidelines.
- `sage_extensions` : Some Extensionf for Chainer Trainer.

# Installation

```bash
$ git clone https://github.com/tn1031/chainer-sagemaker-tools.git
$ cd chainer-sagemaker-tools
$ pip install .
```

When installing to the ML instances, since installing to them is based on the contents of `requirements.txt` , it is necessary to contain the below line on `requirements.txt`.

```
git+https://github.com/tn1031/chainer-sagemaker-tools.git
```

Then put the file in the `source_dir` .

# Run SageMaker training jobs

`smtrain` is a command line tool to run SageMaker training jobs.

### Usage

```bash
$ smtrain <job_name> <path_to_setting> [-p <aws_profile_name>]
```

- `job_name` - Training job name. It must be unique in the same AWS account.
- `path_to_setting` - Path to the setting file. The format of this file is described in [here](https://github.com/tn1031/chainer-sagemaker-tools/blob/master/examples/train.yml).
- `aws_profile_name` - The name of profile that are stored in `~/.aws/config` .

# Deploy trained model

`smdeploy` is a command line tool to deploy.

### Usage

```bash
$ smdeploy <endpoint_name> <path_to_setting> [-p <aws_profile_name>]
```

- `endpoint_name` - Endpoint name.
- `path_to_setting` - Path to the setting file. The format of this file is described in [here](https://github.com/tn1031/chainer-sagemaker-tools/blob/master/examples/deploy.yml).
- `aws_profile_name` - The name of profile that are stored in `~/.aws/config` .

# Run batch inference

`smbatch` is a command line tool to run batch inference.

### Usage

```bash
$ smbatch <model_name> <path_to_setting> [-p <aws_profile_name>]
```

- `model_name` - Model name which used for inference.
- `path_to_setting` - Path to the setting file. The format of this file is described in [here](https://github.com/tn1031/chainer-sagemaker-tools/blob/master/examples/batch.yml).
- `aws_profile_name` - The name of profile that are stored in `~/.aws/config` .
