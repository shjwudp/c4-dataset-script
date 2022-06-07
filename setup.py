# Copyright (c) 2022 Jianbin Chang

from setuptools import find_packages
from setuptools import setup

REQUIRED_PKGS = [
    "tensorflow-datasets",
    "tensorflow",
    "nltk",
    "langdetect",
    "apache_beam",
]

setup(
    name="c4-dataset-script",
    author="Jianbin Chang",
    author_email="shjwudp@gmail.com",
    url="https://github.com/shjwudp/c4-dataset-script",
    license="MIT",
    packages=find_packages(),
    package_data={
        "badwords": [
            "badwords/en",
        ]
    },
    install_requires=REQUIRED_PKGS,
    scripts=[
        "sc4.py"
    ],
    keywords="c4 datasets commoncrawl",
)
