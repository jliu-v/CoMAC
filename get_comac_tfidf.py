#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# get_comac_tfidf.py
# Author: Junfeng Liu
# Created on April 29, 2024 at 22:10
import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BartTokenizer
from transformers import GPT2Tokenizer
import pandas as pd
from data_utils import add_special_tokens_


def get_idf(vocab, dataset, tokenizer):
    corpus = []
    for dialog in dataset['data']:
        # persona entries
        corpus += dialog['persona']
        # knowledge entries
        corpus += dialog['knowledge']
        # full dialog
        last_utterance = dialog['utterance'][-1]
        d_key = [k for k in last_utterance.keys() if k.startswith('dialogue')][0]
        corpus += last_utterance[d_key]

    vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenizer.tokenize, vocabulary=vocab)
    tfidf_weight = vectorizer.fit_transform(corpus)

    tokens = []
    for sk_token, sk_index in vectorizer.vocabulary_.items():
        sk_idf = vectorizer.idf_[sk_index]
        bart_index = tokenizer.convert_tokens_to_ids(sk_token)
        row = (sk_token, sk_index, sk_idf, bart_index)
        tokens.append(row)

    df_tokens = pd.DataFrame(data=tokens, columns=["sk_token", "sk_index", "sk_idf", "bart_index"])
    return df_tokens



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the pre-trained model. {GPT2, BART}")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path of training dataset to generate IDF weights from.")
    parser.add_argument("--output_idf_file_path", type=str, required=True,
                        help="Path of output CSV file of the IDF weights.")
    args = parser.parse_args()

    if args.model_name == "GPT2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif args.model_name == "BART":
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    else:
        raise NotImplementedError("Not supported for model type {}".format(args.model_name))

    add_special_tokens_(model=None, tokenizer=tokenizer)

    # load vocab
    vocab = tokenizer.get_vocab()

    # load training dataset
    with open(args.train_dataset_path, 'r') as fp:
        dataset = json.load(fp=fp)

    # generate idf weights as data frame
    df_tokens = get_idf(vocab=vocab, dataset=dataset, tokenizer=tokenizer)

    # write idf weights to file
    df_tokens.to_csv(args.output_idf_file_path, index=False)


if __name__ == "__main__":
    main()