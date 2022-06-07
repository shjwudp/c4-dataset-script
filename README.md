# C4 Dataset Script

A simple stand-alone C4 dataset production script.

## Getting Started

Setup C4 work environment.

```bash
# 1. Create an independent Anaconda environment and install python dependencies
conda create -y -n c4-env pyspark conda-pack && conda activate c4-env
pip install tensorflow-datasets \
    tensorflow \
    nltk \
    langdetect \
    apache_beam

# 2. Download punkt tokenizer
python -m nltk.downloader -d $(which python | xargs dirname)/../nltk_data punkt

# 3. Run pyspark requires JAVA to be installed in your environment, you should
#    make sure you have JDK installed and JAVA_HOME configured.
```

If everything goes well, you can make the C4 dataset now.

```bash
python sc4.py --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE
```
