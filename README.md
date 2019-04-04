# chainer-sagemaker-tools

This repository is a collection of tools to run SageMaker jobs.

It contains

- `smtrain` : A command line tool to run SageMaker training jobs. See below guidelines.
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

# Run SageMaker training jobs with CLI

`smtrain` is a command line tool to run SageMaker training jobs.

### Usage

```bash
$ smtrain <job_name> <path_to_setting> [-p <aws_profile_name>]
```

- `job_name` - Training job name. It must be unique in the same AWS account.
- `path_to_setting` - Path to the setting file. The format of this file is described in ... [TODO]
- `aws_profile_name` - The name of profile that are stored in `~/.aws/config` .

### Example

TODO
