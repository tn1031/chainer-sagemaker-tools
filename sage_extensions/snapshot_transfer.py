import boto3
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime as dt
from glob import glob
from chainer.training import extension


def snapshot_transfer(keys):
    @extension.make_extension(trigger=(1, 'epoch'), priority=-200)
    def snapshot_transfer(trainer):
        _snapshot_transfer(trainer, keys)

    return snapshot_transfer


def _snapshot_transfer(trainer, keys):
    # [todo] Exception handling
    training_env = os.getenv('SM_TRAINING_ENV')
    module_dir = json.loads(training_env)['module_dir']
    # module_dir: 's3://<bucket_name>/<job_name>/source/sourcedir.tar.gz'
    bucket_name, job_name = module_dir.split('/')[2:4]
    job_name = json.loads(training_env)['job_name']

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    targets = [_get_latest_modified_object(trainer.out, k) for k in keys]

    prefix = 'snapshot' + dt.now().strftime('%y%m%d%H%M') + '_'
    with tempfile.TemporaryDirectory(prefix=prefix, dir=trainer.out) as tmp_path:
        for f in filter(lambda x: x is not None, targets):
            shutil.copyfile(f, os.path.join(tmp_path, os.path.basename(f)))
        out_tar = tmp_path + '.tar.gz'
        with tarfile.open(out_tar, mode='w:gz') as tar:
            tar.add(tmp_path, arcname=os.path.basename(tmp_path))

        dst = os.path.join(job_name, 'snapshot',
                           os.path.basename(out_tar))
        obj = bucket.Object(dst)
        
        try:
            obj.upload_file(out_tar)
        except Exception as e:
            print(e)
        os.remove(out_tar)


def _get_latest_modified_object(dirname, key):
    target = os.path.join(dirname, '%s*' % key)
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return
    latest = sorted(files, key=lambda x: x[1])[-1]
    return latest[0]
