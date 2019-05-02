from setuptools import setup

setup(
    author='tn1031',
    author_email='ttt.nakamura1031@gmail.com',
    name='chainer_sagemaker_tools',
    description='Some extensions and tools to run Chainer jobs on Amazon SageMaker',
    version='0.1.7',
    packages=['sagemaker_tools', 'sage_extensions'],
    install_requires=[
        'boto3',
        'pyyaml>=4.2b1',
        'sagemaker',
        'slackweb'],
    entry_points={
        'console_scripts': ['smtrain=sagemaker_tools.exec_train:main',
                            'smdeploy=sagemaker_tools.deploy_endpoint:main',
                            'smbatch=sagemaker_tools.batch_inference:main'],
    },
    license='MIT license',
)
