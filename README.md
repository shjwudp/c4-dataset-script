# C4 Dataset Script

[C4](https://www.tensorflow.org/datasets/catalog/c4) is a great way to get a colossal cleaned web corpus. Unfortunately, Google open-sourced c4 script highly depends on GCP and code mixed in a big repo. Therefore, it takes work to develop it freely. This repository extracts the processing logic and implements it to run on Spark. In addition, some helpful data process method in MassiveText is implemented in massivetext_utils.py.

## Run c4 script on Spark

Setup c4 work environment.

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
conda pack --name c4-env -o c4-env.tar.gz

# 2. Submit to spark cluster
PYSPARK_DRIVER_PYTHON=python \
PYSPARK_PYTHON=./environment/bin/python \
python c4_dataset_script/c4_script.py \
    --wet-file-paths $PATH_TO_YOUR_CC_WET_FILE \
    --c4-save-path $PATH_TO_YOUR_C4_OUTPUT \
    --spark-master $SPARK_MASTER_ADDR \
    --spark-archives c4-env.tar.gz#environment
```

## Make colossal cleaned Chinese web corpus

Referring to the method of C4, there is a data processing pipeline building for a cleaned Chinese web corpus. It includes web page download, Chinese recognition, heuristics text filter method, toxic recognition and filter, and Repetition Removal used in Google/DeepMind MassiveText.

## 1. Download the WET crawl archive index file

Common Crawl organized crawled data into some archives. You can browse the archives list from [here](https://commoncrawl.org/the-data/get-started/). In the next step, we will download text data (WET) as the input of processing. First, download the WET crawl archive index file.

```bash
cd c4_dataset_script
wget -r --no-parent https://data.commoncrawl.org/crawl-data/${CRAWL_ARCHIVE_ID}/wet.paths.gz
```

*You can get CRAWL_ARCHIVE_ID [here](https://commoncrawl.org/the-data/get-started/). For instance: CC-MAIN-2022-49.*

## 2. Run download and Chinese screening script on Spark

```bash
spark-submit --master ${SPARK_MASTER_ADDR} \
    Chinese/download_web_docs.py \
        --wet-paths ./data.commoncrawl.org/crawl-data/${CRAWL_ARCHIVE_ID}/wet.paths.gz \
        --output ./download-docs
```

## 3. Filter out non-sentence lines and toxic document

Refer to the c4 heuristics method. I used the following strategies for cleaning up Common Crawl's web-extracted text:

 - Only retained lines that ended in a terminal punctuation mark or colon.
 - Discarded any page with fewer than five sentences and only retained lines that
contained at least five words.
 - Removed any page that contained any word on the "List of Dirty, Naughty, Obscene
or Otherwise Bad Words."
 - Many of the scraped pages contained Chinese garbled, so we removed any line with the garbled characters. For example: "[-]|□|■|�".

```bash
cat ./download-docs/*/part-* | \
    python Chinese/filter_out_bad_lines.py \
        --badwords_filepath ./badwords/zh \
         > clean_docs.jsonl
```

*About 93.57% of documents are filtered out in this stage. You can see samples of filtered documents [here](data/Chinese_bad-lines_samples.jsonl).*

## 4. Repetition Removal

Check the percentage of duplicate content in the web document, and the program will remove documents whose duplicate proportion exceeds the preset threshold. This function implements "Repetion Removal" as described in [Gopher](https://arxiv.org/abs/2112.11446).

```bash
spark-submit --master ${SPARK_MASTER_ADDR} \
    Chinese/repetition_removal.py \
        --input clean_docs.jsonl \
        --output ./repetition_removal_output
```

*About 21.21% of documents are filtered out in this stage. You can see samples of filtered documents [here](data/Chinese_Repetition-Removal_samples.jsonl).*
