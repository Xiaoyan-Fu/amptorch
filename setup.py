from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path, listdir
from pkg_resources import DistributionNotFound, get_distribution
from subprocess import check_output
import sys

here = path.abspath(path.dirname(__file__))


# Read version number.

def git_sha():
    try:
        sha = check_output(['git', 'rev-parse', 'HEAD'], cwd=here).decode('ascii').strip()[:7]
    except:
        sha = 'unknown'
    return sha

# required module
# TODO: version check
install_requires = [
    # 'numpy<1.17.0',
    #'scipy<1.3.0',
    # 'scikit-learn<0.21.0',
    'pyyaml>=3.10',
    'cffi>=1.0.0',
    'psutil',
    'tqdm',
    # 'braceexpand',
    'matplotlib<3.0',
]

if sys.version_info >= (3,5):
    install_requires.append('ase>=3.10.0')
else:
    install_requires.append('ase>=3.10.0,<3.18.0')

# Check the differece
setup_requires = [
    'cffi>=1.0.0',
]

def is_installed(pkg):
    try:
        a = get_distribution(pkg)
        return True
    except DistributionNotFound:
        return False


# TODO: install requires add
# FIXME: fill the empty part
setup(
    name='amptorch',
    description='Package for generating atomic potentials using neural network.',
    license='GPL-3.0',
    #keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_data={'':['*.cpp', '*.h']},
    #project_urls={},
    python_requires='>=2.7, <4',
    install_requires=install_requires,
    setup_requires=setup_requires,
    cffi_modules=[
        "amptorch/my_symmetry_function/libsymf_builder.py:ffibuilder"
    ],
)

