"""This script filter out non-sentence lines and toxic text.

```bash
cat docs.jsonl | python filter_out_bad_lines.py --badwords_filepath ../badwords/zh > clean_docs.jsonl
```
"""

import argparse
import sys
import json
import re
import gzip

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Filter out bad lines.")
    parser.add_argument("--badwords_filepath", default=None,
        help="The file path of the toxic word dictionary, if you set this "
        "argument, the program will filter out which document has over limit of"
        " toxic word count. The format of the dictionary file is one word per"
        "line."
    )
    parser.add_argument("--output_bad_lines", default="bad_lines.jsonl.zst",
        help="output file for bad lines")
    parser.add_argument("--bad_words_ratio", default=0.05, type=float,
        help="Document filtering conditions, when the number of bad words in the document exceeds this ratio, it will be screened out.")

    args = parser.parse_args()

    return args


def is_bad_line(line):
    ending_punctuations = ["。", "！", "？", "……", "”", "："]
    if not any(line.endswith(punc) for punc in ending_punctuations):
        return True

    if len(line) < 5:
        return True

    ill_word_regex = "[-]|□|■|�"

    if re.search(ill_word_regex, line) != None:
        return True

    return False


def is_bad_doc(args, doc, badwords_filepath):
    bad_words_character_count = 0
    for bad_word in open(badwords_filepath):
        bad_word = bad_word.strip()
        if bad_word in doc:
            bad_words_character_count += doc.count(bad_word) * len(bad_word)

    if bad_words_character_count / len(doc) > args.bad_words_ratio:
        return True

    return False


def main():
    args = parse_args()
    bad_lines_file = gzip.open(args.output_bad_lines, "wt")

    for line in tqdm(sys.stdin):
        try:
            j = json.loads(line)
        except:
            continue

        if args.badwords_filepath is not None:
            if is_bad_doc(args, j["text"], args.badwords_filepath):
                print(json.dumps(j, ensure_ascii=False), file=bad_lines_file)
                continue

        output = []
        bad_lines = []
        for line in j["text"].splitlines():
            line = line.strip()
            if is_bad_line(line):
                bad_lines.append(line)
            else:
                output.append(line)

        if len(output) > 5:
            j["text"] = '\n'.join(output)
            print(json.dumps(j, ensure_ascii=False))
        else:
            bad_lines += output

        if len(bad_lines) > 0:
            j["text"] = '\n'.join(bad_lines)
            print(json.dumps(j, ensure_ascii=False), file=bad_lines_file)


if __name__ == "__main__":
    main()
