from setuptools import setup

setup(
    author='tn1031',
    author_email='ttt.nakamura1031@gmail.com',
    name='chainer_sagemaker_tools',
    description='Some extensions and tools to run Chainer jobs on Amazon SageMaker',
    version='0.1.0',
    packages=['sagemaker_tools', 'sage_extensions'],
    install_requires=[
        'boto3',
        'PyYAML<4,>=3.10',
        'sagemaker',
        'slackweb'],
    entry_points={
        'console_scripts': ['smtrain=sagemaker_tools.exec_train_on_sm:main'],
    },
    license='MIT license',
)
