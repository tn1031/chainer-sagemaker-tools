import argparse
import os
import yaml
import boto3
from boto3.session import Session

import sagemaker
from sagemaker import get_execution_role
from sagemaker.chainer.estimator import Chainer
from sagemaker.pytorch.estimator import PyTorch


def exec_training(session, client, job_name, setting, pytorch):
    sagemaker_session = sagemaker.Session(
        boto_session=session,
        sagemaker_client=client)

    conf = yaml.load(open(setting))

    # input data
    inputs = conf['inputs']

    if 'upload_data' in conf and isinstance(conf['upload_data'], list):
        for d in conf['upload_data']:
            s3_dir = sagemaker_session.upload_data(
                                  path=d['path'],
                                  key_prefix=os.path.join(job_name, d['key_prefix']))
            inputs[d['name']] = s3_dir

    estimator_args = conf['estimator']
    estimator_args['sagemaker_session'] = sagemaker_session
    if pytorch:
        estimator = PyTorch(**estimator_args)
    else:
        estimator = Chainer(**estimator_args)

    estimator.fit(inputs, job_name=job_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', type=str,
                        help='Training job name. It must be unique.')
    parser.add_argument('setting', type=str,
                        help='Path to setting file.')
    parser.add_argument('--profile_name', '-p', type=str, default=None,
                        help='When execute a training from local, enter the profile name.')
    parser.add_argument('--pytorch', '-t', action='store_true')
    args = parser.parse_args()

    if args.profile_name is None:
        session = Session()
        client = boto3.client('sagemaker', region_name=session.region_name)
    else:
        session = Session(profile_name=args.profile_name)
        credentials = session.get_credentials()

        client = boto3.client('sagemaker', region_name=session.region_name,
                              aws_access_key_id=credentials.access_key,
                              aws_secret_access_key=credentials.secret_key,
                              aws_session_token=credentials.token)

    exec_training(session, client, args.job_name, args.setting, args.pytorch)