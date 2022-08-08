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

"""C4 Dataset Script"""

import collections
import argparse
import re
import functools
import pkg_resources

import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow_datasets.text import c4_utils
import tensorflow_datasets.public_api as tfds


_CITATION = """
@article{2019t5,
  author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {arXiv e-prints},
  year = {2019},
  archivePrefix = {arXiv},
  eprint = {1910.10683},
}
"""


_SENTENCE_TOKENIZER = None


def _load_sentence_tokenizer():
    """Returns a sentence tokenization function."""
    nltk = tfds.core.lazy_imports.nltk
    # Lock to avoid a race-condition in the creation of the download directory.
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def _get_sentences(text):
    global _SENTENCE_TOKENIZER
    if not _SENTENCE_TOKENIZER:
        _SENTENCE_TOKENIZER = _load_sentence_tokenizer()
    return list(_SENTENCE_TOKENIZER.tokenize(tf.compat.as_text(text)))


def get_clean_page_fn():
    """Returns `clean_page` with pre-compiled badword and citation regexes."""
    # Used to filter citation from Wikipedia pages (among others).
    citation_regex = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")
    return functools.partial(clean_page, citation_regex=citation_regex)


def clean_page(url_and_features,
               citation_regex,
               counter_inc_fn=None,
               min_words_per_line=c4_utils._MIN_WORDS_PER_LINE,
               min_num_sentences=c4_utils._MIN_NUM_SENTENCES,
               max_word_length=c4_utils._MAX_WORD_LENGTH):
    """Cleans a CommonCrawl page, yielding nothing if it should be skipped.

    Cleaning removes lines with no end marks or with too few words. After line
    filtering, pages are filtered out if they have too few sentences based on a
    simple count of end marks.

    Args:
      url_and_features: tuple(string, dict), the url and features of the page.
      citation_regex: Regex to use for finding Wikipedia-like citations to filter.
      counter_inc_fn: function, a function taking the name of a counter to be
        incremented and the (optional) amount. Defaults to a beam Metric counter.
      min_words_per_line: int, the minimum number of words a line needs to not be
        removed.
      min_num_sentences: int, the minimum number of sentences a page needs to not
        be skipped.
      max_word_length: int, the maximum number of characters allowed in a word.
        Lines containing a word with too many characters are removed.

    Yields:
      The url and cleaned text for the page.
    """
    url, features = url_and_features
    text = features["text"]

    if not counter_inc_fn:
        counter_inc_fn = c4_utils.get_counter_inc_fn("clean-page")

    lines = text.splitlines()
    valid_lines = []
    num_sentences = 0

    def line_has_too_long_word(line):
        for word in line.split():
            if len(word) > max_word_length:
                return True
        return False

    for line in lines:
        line = line.strip()
        if line_has_too_long_word(line):
            counter_inc_fn("line-filtered:too_long_word")
            continue
        line = citation_regex.sub("", line)
        if not line.endswith(c4_utils._END_MARKS) or line.endswith(c4_utils._ELLIPSIS):
            counter_inc_fn("line-filtered:no_endmark")
            continue
        if len(line.split()) < min_words_per_line:
            counter_inc_fn("line-filtered:too_short")
            continue
        line_lower = line.lower()
        # Remove documents which contain lorem ipsum
        if "lorem ipsum" in line_lower:
            counter_inc_fn("filtered:loremipsum")
            return
        # Remove "javascript must be enabled" notices
        if "javascript" in line_lower:
            counter_inc_fn("line-filtered:javascript")
            continue
        # Remove docs which probably contain javascript code
        if "{" in line:
            counter_inc_fn("filtered:squigglybracket")
            return
        # Remove policy lines
        if any(p in line_lower for p in c4_utils._POLICY_SUBSTRINGS):
            counter_inc_fn("line-filtered:policy")
            continue
        num_sentences += len(_get_sentences(line))
        valid_lines.append(line)
        counter_inc_fn("line-passed")

    if num_sentences < min_num_sentences:
        counter_inc_fn("filtered:too_few_sentences")
        return
    counter_inc_fn("passed")
    features["text"] = "\n".join(valid_lines).strip()
    yield url, features


def _remove_lines_from_text(el, counter_inc_fn, min_num_sentences):
    """Removes all lines from the page that do not match the given set of hashes.

    Process the result of a join containing a single value for 'features' and zero
    or more values for 'lines'. Each value in 'lines' is a lower-cased, hashed
    line that has been selected to keep.

    Args:
      el: `(string, {'features': features_dict, 'lines': [string]})`, element
        containing the result of a join on key with both the page text and
        lower-cased, hashed lines to remove.
      counter_inc_fn: function, a function taking the name of a counter to be
        incremented and the (optional) amount.
      min_num_sentences: int, the minimum number of sentences a page needs to not
        be skipped.

    Yields:
      url: The URL of the page.
      features: The page features with lines removed from text.
    """
    url, join_values = el
    features = join_values["features"]

    assert len(features) == 1, "Invalid page count (%d) for %s" % (len(features),
                                                                   url)
    features = features[0]
    text = features["text"]
    lines_to_keep = set(join_values["lines"])
    new_lines = []
    hashed_lines = set()
    for line in text.split("\n"):
        hashed_line = c4_utils._hash_text(line.strip().lower())
        if hashed_line not in lines_to_keep:
            counter_inc_fn("line-filtered:global_duplicate")
        elif hashed_line in hashed_lines:
            counter_inc_fn("line-filtered:local_duplicate")
        else:
            counter_inc_fn("line-passed")
            new_lines.append(line)
            hashed_lines.add(hashed_line)
    new_text = "\n".join(new_lines)
    if not new_text:
        counter_inc_fn("filtered:empty")
        return
    if min_num_sentences and len(_get_sentences(new_text)) < min_num_sentences:
        counter_inc_fn("filtered:too_few_sentences")
        return
    counter_inc_fn("passed")
    new_features = features.copy()
    new_features["text"] = new_text
    yield (url, new_features)


def remove_duplicate_text(pages, min_num_sentences=c4_utils._MIN_NUM_SENTENCES):
    """Utility to remove duplicate lines across text documents."""
    # Output: url, lines

    # Select a single URL for each line in the input pages.
    # Hash before comparison to avoid biasing by domain.
    # line, url
    line_to_selected_url = pages.flatMap(c4_utils._emit_url_to_lines)\
        .reduceByKey(lambda a, b: a)

    # Transform to url, [line]
    lines_to_keep = line_to_selected_url.map(lambda x: (x[1], x[0]))\
        .groupByKey()\
        .mapValues(list)

    # Output: url, text
    final_docs = pages.join(lines_to_keep)\
        .mapValues(lambda x: {"features": list(x[0]), "lines": x[1]})\
        .flatMap(lambda x: _remove_lines_from_text(list(x), counter_inc_fn=c4_utils.get_counter_inc_fn("dedupe-lines"), min_num_sentences=min_num_sentences))

    return final_docs


def dedupe_urls(a, b):
    hash_a = c4_utils._hash_text(a["text"])
    hash_b = c4_utils._hash_text(b["text"])

    if hash_a > hash_b:
        return a
    return b


def c4_process(args):
    if args.spark_archives:
        spark = SparkSession.builder.config("spark.archives", args.spark_archives)\
            .master(args.spark_master)\
            .getOrCreate()
    else:
        spark = SparkSession.builder.master(args.spark_master).getOrCreate()

    wet_file_paths = spark.sparkContext.parallelize(args.wet_file_paths)

    page_content = wet_file_paths\
        .flatMap(c4_utils.split_wet_file)\
        .filter(c4_utils.is_valid_length)\
        .map(c4_utils.normalize_url)\
        .reduceByKey(dedupe_urls)

    if args.paragraph_filter:
        page_content = page_content.filter(c4_utils.paragraph_filter)

    if args.clean:
        page_content = page_content.flatMap(get_clean_page_fn())

    if args.dedupe:
        page_content = remove_duplicate_text(page_content)

    page_content = page_content.flatMap(c4_utils.detect_english)

    if args.badwords_filter:
        # Create dictionary of badwords regex for each available language.
        badwords = collections.defaultdict(set)
        lang = "en"
        path = args.badwords_file_path
        with tf.io.gfile.GFile(path) as f:
            badwords[lang].update(l.strip() for l in f)

        page_content = page_content.filter(c4_utils.get_badwords_filter_fn(badwords))

    return page_content


def parse_args():
    parser = argparse.ArgumentParser(
        description="C4 Dataset Manufacturing Script")

    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--spark-archives", default=None,
                        help="https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html#using-conda")
    parser.add_argument("--wet-file-paths", nargs='+')
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

    args = parser.parse_args()

    if args.badwords_file_path is None:
        args.badwords_file_path = pkg_resources.resource_filename("c4_dataset_script", "badwords/en")

    return args


def main():
    args = parse_args()

    c4_text = c4_process(args)
    c4_text.saveAsTextFile(args.c4_save_path)


if __name__ == "__main__":
    main()
