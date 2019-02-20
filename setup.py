from setuptools import setup

setup(
    name='teebee',
    version='0.1.1',
    url='https://github.com/sublee/teebee',
    license='MIT',
    author='Heungsub Lee',
    author_email='sub@subl.ee',
    description='1k steps in TensorBoard for 1 epoch.',
    zip_safe=False,
    setup_requires=['tensorboardX'],
    py_modules=['teebee'],
)
