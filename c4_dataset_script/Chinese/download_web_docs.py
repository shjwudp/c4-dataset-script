# Copyright (c) 2022 Jianbin Chang

import argparse
import json
import gzip
import io
import logging
import time
import os

from pyspark.sql import SparkSession
import langdetect
import requests
from tqdm import tqdm


CC_DOMAIN = "https://data.commoncrawl.org"

# WET file constants
_PAGE_DELIMITER = "WARC/1.0"
_URL_KEY = "WARC-Target-URI:"
_URL_DATE = "WARC-Date:"
_CONTENT_TYPE = "Content-Type:"
_CONTENT_LANGUAGE = "WARC-Identified-Content-Language:"
_METADATA_PREFIXES = ("WARC", "CONTENT-", "Content-")


def check_if_gz_file_corrupted(gz_file):
    chunksize = 10 * 1024 ** 2

    with gzip.open(gz_file, 'rb') as f:
        try:
            while f.read(chunksize) != b'':
                pass
            return False
        except:
            return True


def split_wet_file(wet_file_path):
    def _validate_features(page):
        feature_list = ["url", "text", "timestamp"]
        for feature_name in feature_list:
            if feature_name not in page:
                return False

        return True

    page = dict()
    for i, line in enumerate(gzip.open(wet_file_path, "rt")):
        line = line.strip()
        if not line:
            continue

        if line == _PAGE_DELIMITER:
            if i > 0 and _validate_features(page):
                yield page
            page = dict()

        if line.startswith(_URL_KEY):
            page["url"] = line[len(_URL_KEY):].strip()

        if line.startswith(_URL_DATE):
            page["timestamp"] = line[len(_URL_DATE):].strip()

        if line.startswith(_CONTENT_TYPE):
            page["content_type"] = line[len(_CONTENT_TYPE):].strip()

        if line.startswith(_CONTENT_LANGUAGE):
            page["content_language"] = line[len(_CONTENT_LANGUAGE):].strip()

        if line.startswith(_METADATA_PREFIXES):
            continue

        if "text" in page:
            page["text"] += "\n"
        page["text"] = page.get("text", "") + line

    if _validate_features(page):
        yield page


def request_with_retry(connection_reset_retry=20, *args, **kwargs):
    retries = 0
    while True:
        try:
            response = requests.get(*args, **kwargs, timeout=3600)
            return response
        except (
            ConnectionResetError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            if retries >= connection_reset_retry:
                logging.info(f"{args}")
                raise
            time.sleep(2 ** retries)
            retries += 1


def download_and_package(
    cc_path,
    chinese_filtering=True,
):
    logging.basicConfig(level=logging.DEBUG)

    for _ in range(10):
        response = request_with_retry(url=f"{CC_DOMAIN}/{cc_path}")
        download_file = io.BytesIO(response.content)
        page_list = []
        try:
            for page in tqdm(split_wet_file(download_file), desc=f"split_wet_file {download_file}"):
                if chinese_filtering:
                    if "content_language" not in page:
                        try:
                            language = langdetect.detect(page["text"])
                        except langdetect.lang_detect_exception.LangDetectException:
                            continue
                        if language not in ["zh-cn", "zh-tw"]:
                            continue
                    elif "zho" not in page["content_language"].split(","):
                        continue

                page_list.append(page)
            break
        except (EOFError, gzip.BadGzipFile):
            continue

    for page in page_list:
        yield page


def read_wet_paths_file(filepath):
    for line in gzip.open(filepath, "rt"):
        cc_path = line.strip()
        yield cc_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet-paths", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spark-sub-job", default=50,
        help="From the data dimention, divide the spark job into sub-jobs, reducing the loss of job failed.")
    args = parser.parse_args()

    spark = SparkSession.builder\
            .appName("Download Chinese web docs")\
            .getOrCreate()

    cc_paths = []
    for wet_path in args.wet_paths:
        for cc_path in read_wet_paths_file(wet_path):
            cc_paths.append(cc_path)

    for i in range(args.spark_sub_job):
        batch_size = len(cc_paths) // args.spark_sub_job + 1
        input =  cc_paths[i * batch_size: (i + 1) * batch_size]
        output_dir = os.path.join(args.output, str(i))
        rdd = spark.sparkContext.parallelize(input)\
            .repartition(128)\
            .flatMap(lambda cc_path: download_and_package(cc_path, args.output))\
            .map(lambda page: json.dumps(page, ensure_ascii=False))

        rdd.saveAsTextFile(output_dir)


if __name__ == "__main__":
    main()
