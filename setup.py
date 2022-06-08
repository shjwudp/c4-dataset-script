from setuptools import find_packages
from setuptools import setup

REQUIRED_PKGS = [
    "pyspark",
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
        "c4_dataset_script": [
            "badwords/en",
        ]
    },
    install_requires=REQUIRED_PKGS,
    scripts=[
        "c4_dataset_script/c4_script.py",
    ],
    keywords="c4 datasets commoncrawl",
)
