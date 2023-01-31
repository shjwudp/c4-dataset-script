import argparse
import nltk
import hashlib
import os
import json

from pyspark.sql import SparkSession
import jieba


def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument("--output_bad_docs", default="bad_docs.jsonl.zst",
        help="output file for bad lines")

    args = parser.parse_args()

    return args


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_repetition_removal(
    text, duplicate_line_fraction=0.3, duplicate_line_character_faction=0.2
):
    """Check if there is repeated content in the input text. Excessive
    repetition is often linked with uninformative content and can be used to
    determine whether it is low-quality text. This function implements
    "Repetition Removal" as described in Gopher_.

    .. _Gopher: https://arxiv.org/abs/2112.11446

    Args:
        text (str): input text.
        duplicate_line_fraction (float, optional): Duplicate row deduplication
            threshold. Defaults to 0.3.
        duplicate_line_character_faction (float, optional): Threshold for the
            proportion of repeated line characters. Defaults to 0.2.

    Returns:
        bool: If there is repeated content in the input text.
    """
    line_count = 0
    dup_line = 0
    dup_line_chars = 0
    visit_lines = {}
    for line in text.split("\n"):
        line_hash = hash_text(line)
        if line_hash in visit_lines:
            dup_line += 1
            dup_line_chars += len(line)
        visit_lines[line_hash] = True

        line_count += 1

    if float(dup_line) / line_count > duplicate_line_fraction:
        return True

    if float(dup_line_chars) / len(text) > duplicate_line_character_faction:
        return True

    top_ngram_character_fractions = [
        (2, 0.2),
        (3, 0.18),
        (4, 0.16),
    ]
    for ngram, threshold in top_ngram_character_fractions:
        word_list = list(jieba.cut(text))
        bgs = nltk.ngrams(word_list, ngram)
        fdist = nltk.FreqDist(bgs)
        for word_list, repeat in fdist.items():
            char_count = sum([len(word) for word in word_list])
            if char_count * (repeat - 1) / len(text) > threshold:
                return True

    duplicate_ngram_character_fractions = [
        (5, 0.15),
        (6, 0.14),
        (7, 0.13),
        (8, 0.12),
        (9, 0.11),
        (10, 0.10),
    ]
    for ngram, threshold in duplicate_ngram_character_fractions:
        fdist = {}
        word_list = list(jieba.cut(text))
        mark = [0] * len(word_list)
        for i in range(len(word_list) - ngram + 1):
            bag = tuple(word_list[i: i + ngram])
            if bag in fdist:
                for j in range(i, i + ngram):
                    mark[j] = len(word_list[j])
                fdist[bag] += 1
            else:
                fdist[bag] = 1

        if sum(mark) / float(len(text)) > threshold:
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="./rr_output")
    args = parser.parse_args()

    spark = SparkSession.builder\
            .appName("Repetional Removal")\
            .getOrCreate()

    docs = spark.sparkContext.textFile(args.input)\
        .repartition(64)\
        .map(lambda line: json.loads(line))\
        .map(lambda doc: (doc, is_repetition_removal(doc["text"])))

    clean_docs = docs.filter(lambda x: not x[1])\
        .map(lambda x: json.dumps(x[0], ensure_ascii=False))

    bad_docs = docs.filter(lambda x: x[1])\
        .map(lambda x: json.dumps(x[0], ensure_ascii=False))

    clean_docs.saveAsTextFile(os.path.join(args.output, "clean_docs"))
    bad_docs.saveAsTextFile(os.path.join(args.output, "bad_docs"))


if __name__ == "__main__":
    main()
