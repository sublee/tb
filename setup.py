"""teebee: 1k steps for 1 epoch in TensorBoard"""
from pathlib import Path

from setuptools import setup

about = {}  # type: ignore
with (Path('teebee')/'__about__.py').open() as f:
    exec(f.read(), about)  # pylint: disable=W0122


setup(
    name='teebee',
    version=about['__version__'],
    url=about['__url__'],
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    zip_safe=False,
    setup_requires=['tensorboardX'],
    package_data={'teebee': ['py.typed']},
    packages=['teebee'],
)
