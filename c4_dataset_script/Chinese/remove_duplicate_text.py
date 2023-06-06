"""
```bash
spark-submit \
    --master "local[*]" \
    Chinese/remove_duplicate_text.py \
        --input clean_docs.jsonl
```
"""

import argparse
import hashlib
import os
import json

from pyspark.sql import SparkSession


MIN_NUM_SENTENCES = 5


def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument("--output_bad_docs", default="bad_docs.jsonl.zst",
        help="output file for bad lines")

    args = parser.parse_args()

    return args


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _remove_lines_from_text(el, min_num_sentences):
    """Removes all lines from the page that do not match the given set of
    hashes.
    Process the result of a join containing a single value for 'features' and
    zero or more values for 'lines'. Each value in 'lines' is a lower-cased,
    hashed line that has been selected to keep.
    Args:
      el: `(string, {'features': features_dict, 'lines': [string]})`, element
        containing the result of a join on key with both the page text and
        lower-cased, hashed lines to remove.
      counter_inc_fn: function, a function taking the name of a counter to be
        incremented and the (optional) amount.
      min_num_sentences: int, the minimum number of sentences a page needs to
        not be skipped.
    Yields:
      url: The URL of the page.
      features: The page features with lines removed from text.
    """
    url, join_values = el
    features = join_values["features"]
    text = features["text"]
    lines_to_keep = set(join_values["lines"])
    new_lines = []
    hashed_lines = set()
    for line in text.split("\n"):
        hashed_line = hash_text(line.strip().lower())
        if hashed_line not in lines_to_keep:
            pass
        elif hashed_line in hashed_lines:
            pass
        else:
            new_lines.append(line)
            hashed_lines.add(hashed_line)
    new_text = "\n".join(new_lines)
    if not new_text:
        return
    if min_num_sentences and len(new_text.splitlines()) < min_num_sentences:
        return
    new_features = features.copy()
    new_features["text"] = new_text
    yield (url, new_features)


def remove_duplicate_text(docs, min_num_sentences=MIN_NUM_SENTENCES):
    """Utility to remove duplicate lines across text documents."""
    # Output: url, lines

    docs = docs.map(lambda doc: (doc["url"], doc))

    # Select a single URL for each line in the input docs.
    # Hash before comparison to avoid biasing by domain.
    def emit_url_to_lines(doc):
        # => (line_hash, (url, line))
        for line in doc["text"].splitlines():
            yield hash_text(line.strip().lower()), (doc["url"], line)

    def merge_duplicate_line_group(a, b):
        if hash_text(a[0]) > hash_text(b[0]):
            a, b = b, a

        a[-1] += b[-1]
        return a

    line_to_selected_url = docs\
        .flatMap(lambda url_doc: emit_url_to_lines(url_doc[1]))\
        .mapValues(lambda x: list(x) + [1])\
        .reduceByKey(lambda a, b: merge_duplicate_line_group(a, b))

    removed_lines = line_to_selected_url.filter(lambda x: x[1][-1] > 1)\
        .map(lambda x: x[1])

    # (line_hash, (url, line, repeat)) => (url, line_hash)
    lines_to_keep = line_to_selected_url.map(lambda x: (x[1][0], x[0]))

    # Modifications effective on the original text, remove document with less
    # than the preset num of sentences.
    final_docs = docs.cogroup(lines_to_keep, numPartitions=1024)\
        .mapValues(lambda x: {"features": list(x[0])[0], "lines": list(x[1])})\
        .flatMap(lambda x: _remove_lines_from_text(list(x), min_num_sentences=min_num_sentences))\
        .map(lambda x: x[1])

    return final_docs, removed_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="./rd_output")
    args = parser.parse_args()

    spark = SparkSession.builder\
            .appName("Remove duplicate text")\
            .getOrCreate()

    docs = spark.sparkContext.textFile(args.input)\
        .sample(withReplacement=False, fraction=1.0)\
        .repartition(512)\
        .map(lambda line: json.loads(line))

    clean_docs, removed_lines = remove_duplicate_text(docs)

    clean_docs = clean_docs.map(lambda j: json.dumps(j, ensure_ascii=False))
    removed_lines = removed_lines.sample(withReplacement=False, fraction=0.01)

    clean_docs.saveAsTextFile(os.path.join(args.output, "clean_docs"))
    removed_lines.saveAsTextFile(os.path.join(args.output, "removed_lines"))


if __name__ == "__main__":
    main()
