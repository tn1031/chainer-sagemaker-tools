import argparse
import os
import yaml
import boto3
from boto3.session import Session
from datetime import datetime as dt

import sagemaker
from sagemaker.chainer.model import ChainerModel
from sagemaker.pytorch.model import PyTorchModel


def deploy_endpoint(session, client, model_name, setting, pytorch):
    sagemaker_session = sagemaker.Session(
        boto_session=session,
        sagemaker_client=client)

    conf = yaml.load(open(setting))

    # is model uploader required?

    model_args = conf['model']
    model_args['sagemaker_session'] = sagemaker_session
    model_args['name'] = model_name
    if pytorch:
        model = PyTorchModel(**model_args)
    else:
        model = ChainerModel(**model_args)

    deploy_args = conf['deploy']
    transformer = model.transformer(**deploy_args)  # register model

    transform_args = conf['transform']
    # transform_args['job_name'] = model_name + datetime.now()
    transformer.transform(**transform_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str,
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

    deploy_endpoint(session, client, args.model_name, args.setting, args.pytorch)