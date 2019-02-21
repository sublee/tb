"""teebee: 1k steps in TensorBoard for 1 epoch"""
from setuptools import setup


setup(
    name='teebee',
    version='0.1.3',
    url='https://github.com/sublee/teebee',
    license='MIT',
    author='Heungsub Lee',
    author_email='sub@subl.ee',
    description='1k steps in TensorBoard for 1 epoch.',
    zip_safe=False,
    setup_requires=['tensorboardX'],
    package_data={'teebee': ['py.typed']},
    packages=['teebee'],
)
