# C4 Dataset Script

A simple stand-alone C4 dataset production script.

## Getting Started

Setup C4 work environment.

```bash
# 1. Create an independent Anaconda environment and install python dependencies
conda create -y -n c4-env conda-pack && conda activate c4-env
pip install git+https://github.com/shjwudp/c4-dataset-script

# 2. Download punkt tokenizer
python -m nltk.downloader -d $(which python | xargs dirname)/../nltk_data punkt

# 3. Run pyspark requires JAVA to be installed in your environment, you should
#    make sure you have JDK installed and JAVA_HOME configured.
```

If everything goes well, you can make the C4 dataset on localhost.

```bash
python -m c4_dataset_script.c4_script --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE
```

Or submit to spark cluster.

```bash
# 1. Before submitting to the cluster, you need to package the environment conda env
conda pack -name c4-env -o c4-env.tar.gz

# 2. Submit to spark cluster
PYSPARK_DRIVER_PYTHON=python \
PYSPARK_PYTHON=./environment/bin/python \
python c4_dataset_script/c4_script.py \
    --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE \
    --c4-save-path $PATH_TO_YOUR_C4_OUTPUT \
    --spark-master $SPARK_MASTER_ADDR \
    --spark-archives c4-env.tar.gz#environment
```
