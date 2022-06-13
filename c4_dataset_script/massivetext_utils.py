import nltk
import hashlib

from pyspark.ml import Pipeline
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH
from pyspark.sql import functions as F


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_repetition_removal(
    text, duplicate_line_fraction=0.3, duplicate_line_character_faction=0.2
):
    """Check if there is repeated content in the input text. Excessive repetition
    is often linked with uninformative content, can be be used to determine whether
    it is low-quality text. This function implements "Repetition Removal" as
    described in Gopher_,

    .. _Gopher: https://arxiv.org/abs/2112.11446

    Args:
        text (str): input text.
        duplicate_line_fraction (float, optional): Duplicate row deduplication threshold.
            Defaults to 0.3.
        duplicate_line_character_faction (float, optional): Threshold for the proportion
            of repeated line characters. Defaults to 0.2.

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
        bgs = nltk.ngrams(text.split(), ngram)
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
        word_list = text.split()
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


def docs_dedup(docs, ngram=13, jaccard_similarity_threshold=0.8, min_hash_tables=20):
    """Use the MinHash algorithm to calculate all approximately duplicate documents and
    remove them, obtain a set of unique documents. This function implements "Document Deduplication"
    as described in Gopher_,

    .. _Gopher: https://arxiv.org/abs/2112.11446

    Args:
        docs (pyspark.sql.DataFrame): Document text and id, schema ("id", "text").
        ngram (int, optional): NGram for text vectorization. Defaults to 13.
        jaccard_similarity_threshold (float, optional): Jaccard similarities threshold. Defaults to 0.8.
        min_hash_tables (int, optional): Number of MinHash hash tables. Defaults to 20.
    """

    def tokenize(docs_partition):
        # Ignore whitespace and punctuation.
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        for doc in docs_partition:
            tokens = tokenizer.tokenize(doc["text"])
            if len(tokens) < ngram:
                continue
            yield doc["id"], hash_text(doc["id"]), tokens

    tokens = docs.rdd.mapPartitions(tokenize).toDF(["id", "id_hash", "tokens"])
    model = Pipeline(stages=[
        NGram(n=ngram, inputCol="tokens", outputCol="ngrams"),
        HashingTF(inputCol="ngrams", outputCol="vectors"),
        MinHashLSH(inputCol="vectors", outputCol="lsh",
                   numHashTables=min_hash_tables),
    ]).fit(tokens)

    tokens_hashed = model.transform(tokens)
    duplicate_pairs = model.stages[-1].approxSimilarityJoin(
        tokens_hashed, tokens_hashed, 1 - jaccard_similarity_threshold
    ).select(
        F.col("datasetA.id_hash").alias("A_hash"),
        F.col("datasetB.id_hash").alias("B_hash"),
    ).filter("A_hash < B_hash")

    deduplicated_items = tokens.join(duplicate_pairs, tokens.id_hash == duplicate_pairs.A_hash, how="left_anti")\
        .select(F.col("id"))

    tokens_A = tokens.alias("A")
    tokens_B = tokens.alias("B")
    duplicate_pairs_id = duplicate_pairs.join(tokens_A, duplicate_pairs.A_hash == tokens_A.id_hash, how="left")\
        .join(tokens_B, duplicate_pairs.B_hash == tokens_B.id_hash, how="left")\
        .select(
            F.col("A.id").alias("A_id"),
            F.col("B.id").alias("B_id"),
    )

    return deduplicated_items, duplicate_pairs_id
