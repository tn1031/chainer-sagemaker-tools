import boto3
import json
import hashlib
import os
import slackweb
from chainer.training import extension

def slack_report(keys, hook, channel, pretext=None,
                 public_bucket_name=None, region=None):
    @extension.make_extension(trigger=(1, 'epoch'), priority=0)
    def slack_report(trainer):
        _slack_report(trainer, keys, hook, channel, pretext,
                      public_bucket_name, region)

    return slack_report

def _slack_report(trainer, keys, hook, channel, pretext,
                  public_bucket_name, region):
    slack = slackweb.Slack(url=hook)

    log_report = trainer.get_extension('LogReport')
    current_log = log_report.log[-1]

    attachments = list()

    fields = list()
    color = 'good'
    for k in keys:
        if k in current_log:
            fields.append({'title': k, 'value': current_log[k], 'short': True})
            if not (isinstance(current_log[k], float) or \
                    isinstance(current_log[k], int)):
                color = 'danger'

    attachments.append(
        {'pretext': pretext, 'color': color, 'fields': fields})

    plot_reports = [v.extension for k, v in trainer._extensions.items() if 'PlotReport' in k]
    if len(plot_reports) > 0 and public_bucket_name is not None:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(public_bucket_name)  # [todo] Exception handling

        for pr in plot_reports:
            name = pr._file_name
            if os.path.isfile(os.path.join(trainer.out, name)):
                try:
                    url = _upload_figure(name, trainer.out, bucket,
                                         public_bucket_name, region)
                    attachments.append({'color': color,
                                        'image_url': url,
                                        'fields': [{'value': name}]})
                except:
                    attachments.append({'color': 'danger',
                                        'text': 'image uploading was failed.'})

    try:
        training_env = os.getenv('SM_TRAINING_ENV')
        job_name = json.loads(training_env)['job_name']
    except:
        job_name = os.uname()[1]
        
    slack.notify(channel=channel,
        text='Training Report from %s' % job_name,
        attachments=attachments)  # [todo] Exception handling

def _upload_figure(name, out, bucket, public_bucket_name, region):
    path = os.path.join(out, name)

    with open(path, 'rb') as f:
        chsum = hashlib.md5(f.read()).hexdigest()
    dst = chsum + '.png'
    obj = bucket.Object(dst)
    obj.upload_file(path)

    url = 'https://s3-{}.amazonaws.com/{}/{}'.format(region, public_bucket_name, dst)
    return url
