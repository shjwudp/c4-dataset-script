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
    parser.add_argument("--badwords_filepath", required=True)
    parser.add_argument("--output_bad_lines", default="bad_lines.jsonl.zst", help="output file for bad lines")

    args = parser.parse_args()

    return args


def is_bad_line(line):
    ending_punctuations = ["。", "！", "？", "……", "”", "："]
    if not any(line.endswith(punc) for punc in ending_punctuations):
        return True

    if len(line) < 5 or len(line) > 500:
        return True

    ill_word_regex = "[-]|□|■|[①-⑳]|[⑴-⒇]|[㈠-㈩]|[⒈-⒓]"

    if re.search(ill_word_regex, line) != None:
        return True

    return False


def is_bad_doc(doc, badwords_filepath):
    count = 0
    for bad_word in open(badwords_filepath):
        bad_word = bad_word.strip()
        if bad_word in doc:
            count += doc.count(bad_word)
            if count > 3:
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

        if is_bad_doc(j["text"], args.badwords_filepath):
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
