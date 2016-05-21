import os
from setuptools import find_packages
from setuptools import setup

version = '0.1.dev1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'lasagne'
    # 'Theano',  # we require a development version, see requirements.txt
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
    ]

setup(
    name="Britefury Lasagne Helpers",
    version=version,
    description="Helpers and tools for simplifying the construction and training of deep neural networks using Lasagne",
    long_description="\n\n".join([README]),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        # "Programming Language :: Python :: 3",
        # "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Geoffrey French",
    # author_email="brittix1023 at gmail dot com",
    url="https://github.com/Britefury/britefury-lasagne-helpers",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
