import argparse
import os
import yaml
import boto3
from boto3.session import Session

import sagemaker
from sagemaker import get_execution_role
from sagemaker.chainer.estimator import Chainer
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tuner import HyperparameterTuner, CategoricalParameter, \
                            ContinuousParameter, IntegerParameter
hp_type = {'continuous': ContinuousParameter,
           'integer': IntegerParameter,
           'categorical': CategoricalParameter}

def exec_training(session, client, job_name, setting, pytorch, max_parallel_jobs):
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

    hyperparameters = estimator_args.pop('hyperparameters')
    fixed, targets = {}, {}
    for k, v in hyperparameters.items():
        if isinstance(v, dict):
            targets[k] = v
        else:
            fixed[k] = v
    estimator_args['hyperparameters'] = fixed

    if pytorch:
        estimator = PyTorch(**estimator_args)
    else:
        estimator = Chainer(**estimator_args)

    if len(targets) == 0:
        estimator.fit(inputs, job_name=job_name)
    else:
        if 'tuner' in conf:
            tuner_args = conf['tuner']
            hyperparameter_ranges = {}
            for k, v in targets.items():
                hyperparameter_ranges[k] = hp_type[v['type'].lower()](v['range'])
        else:  # use default values
            tuner_args = {'objective_metric_name': 'metric_name',
                          'metric_definitions': [{'Name': 'metric_name', 'Regex': 'ignore'}],
                          'strategy': 'Random',
                          'objective_type': 'Maximize',
                          'early_stopping_type': 'Off'}
            max_jobs = 1
            hyperparameter_ranges = {}
            for k, v in targets.items():
                if v['type'].lower() != 'categorical':
                    raise ValueError('the default tuner only supports Categorigal params.')
                max_jobs *= len(v['range'])
                hyperparameter_ranges[k] = hp_type[v['type'].lower()](v['range'])
            tuner_args['max_jobs'] = max_jobs

        tuner_args['estimator'] = estimator
        tuner_args['hyperparameter_ranges'] = hyperparameter_ranges
        tuner_args['max_parallel_jobs'] = max_parallel_jobs
        tuner_args['base_tuning_job_name'] = job_name
        tuner_args['warm_start_config'] = None  # not supported yet.

        tuner = HyperparameterTuner(**tuner_args)
        tuner.fit(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', type=str,
                        help='Training job name. It must be unique.')
    parser.add_argument('setting', type=str,
                        help='Path to setting file.')
    parser.add_argument('--profile_name', '-p', type=str, default=None,
                        help='When execute a training from local, enter the profile name.')
    parser.add_argument('--pytorch', '-t', action='store_true')
    parser.add_argument('--max_parallel_jobs', type=int, default=1,
                        help='# wokers for bulk training.')
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

    exec_training(session, client, args.job_name, args.setting, args.pytorch,
                  args.max_parallel_jobs)