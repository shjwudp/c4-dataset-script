# Copyright (c) 2022 Jianbin Chang
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data cleaning methods mentioned in WeLM_.

.. _WeLM: https://arxiv.org/abs/2209.10372
"""

import argparse
import os
import uuid

import pyspark
import langdetect
import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow_datasets.text import c4_utils
import tensorflow_datasets.public_api as tfds


_DOWNLOAD_HOST = "https://data.commoncrawl.org"

def dedupe_urls(a, b):
    hash_a = c4_utils._hash_text(a["text"])
    hash_b = c4_utils._hash_text(b["text"])

    if hash_a > hash_b:
        return a
    return b


def download_wet_file(path, dl_dir):
    url = f"{_DOWNLOAD_HOST}/{path}"
    out_path = f"{dl_dir}/{path}"

    if tf.io.gfile.exists(out_path):
        c4_utils.get_counter_inc_fn("download_wet_url")("exists")
        return out_path

    tmp_dir = f"{out_path}.incomplete{uuid.uuid4().hex}"
    try:
        tf.io.gfile.makedirs(tmp_dir)
        downloader = tfds.download.downloader.get_downloader()
        with downloader.tqdm():
            # TODO(slebedev): Investigate why pytype infers Promise[Future[...]].
            dl_path = downloader.download(url, tmp_dir).get().path  # type: ignore
        tf.io.gfile.rename(os.fspath(dl_path), out_path, overwrite=True)
    finally:
        if tf.io.gfile.exists(tmp_dir):
            tf.io.gfile.rmtree(tmp_dir)

    c4_utils.get_counter_inc_fn("download_wet_url")("downloaded")
    return out_path


def c4_process(args):
    if args.spark_archives:
        spark = SparkSession.builder.config("spark.archives", args.spark_archives)\
            .master(args.spark_master)\
            .getOrCreate()
        sc = spark.sparkContext
    elif args.spark_master == "gcp":
        sc = pyspark.SparkContext()
    else:
        spark = SparkSession.builder.master(args.spark_master).getOrCreate()
        sc = spark.sparkContext
    sc.setLogLevel(args.spark_log_level)

    wet_file_paths = sc.textFile(args.wet_file_paths)\
        .repartition(args.input_repartition)

    def filter_chinese_content(url_doc):
        text = url_doc[1]["text"]
        try:
            return langdetect.detect(text) in ["zh-cn", "zh-tw"]
        except:
            return False

    page_content = wet_file_paths\
        .map(lambda wet_path: download_wet_file(wet_path, os.path.join(args.download_dir, "c4_wet_files")))\
        .flatMap(c4_utils.split_wet_file)\
        .filter(c4_utils.is_valid_length)\
        .filter(filter_chinese_content)\
        .map(c4_utils.normalize_url)\
        .reduceByKey(dedupe_urls)

    return page_content


def parse_args():
    parser = argparse.ArgumentParser(
        description="C4 Dataset Manufacturing Script")

    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--spark-archives", default=None,
                        help="https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-conda")
    parser.add_argument("--wet-file-paths", required=True)
    parser.add_argument("--download-dir", required=True, help="Download file directory, you can configure shared storage.")
    parser.add_argument("--input-repartition", default=1000)
    parser.add_argument("--c4-save-path", default="./c4")
    parser.add_argument('--no-paragraph-filter', action='store_false',
                       help='Do not filter paragraph.',
                       dest='paragraph_filter')
    parser.add_argument('--no-clean', action='store_false',
                       help='Do not remove lines with no end marks or with too few words.',
                       dest='clean')
    parser.add_argument('--no-dedupe', action='store_false',
                       help='Do not dedupelicate lines across text documents.',
                       dest='dedupe')
    parser.add_argument('--no-badwords-filter', action='store_false',
                       help='Do not filter out pages that contain any language-specific bad words.',
                       dest='badwords_filter')
    parser.add_argument("--badwords-file-path", type=str, default=None)
    parser.add_argument("--spark-log-level", default="ERROR", choices=["ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"])

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    c4_text = c4_process(args)
    c4_text.saveAsTextFile(args.c4_save_path)


if __name__ == "__main__":
    main()
