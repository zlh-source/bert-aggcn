"""
Prepare vocabulary and initial word vectors.
"""
import torch
import numpy as np
import json
import pickle
from collections import Counter
from pytorch_pretrained_bert import BertTokenizer, BertModel


def main():
    print("# Load pre-trained model tokenizer (vocabulary)")
    tokenizer = BertTokenizer.from_pretrained('./dataset/bert/')

    print("# Construct vocab")
    vocabulary = [token for token in tokenizer.vocab]
    vocab = set(vocabulary)
    print("Vocabulary Size: {}".format(len(vocabulary)))

    print("# Load pre-trained model")
    model = BertModel.from_pretrained('./dataset/bert/')

    print("# Load word embeddings")
    emb = model.embeddings.word_embeddings.weight.data
    emb = emb.numpy()
    print("# Embedding size: {} x {}".format(*emb.shape))

    train_file = "./dataset/trp/data/train.json" 
    dev_file = "./dataset/trp/data/dev.json"
    test_file = "./dataset/trp/data/test.json"

    vocab_file = "./dataset/trp/vocab/vocab.pkl"
    emb_file = "./dataset/trp/vocab/embedding.npy"

    print("# Loading TrP files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)

    v = build_vocab(train_tokens, vocab, 0)

    print("# Calculating TrP oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
    print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))
    
    print("# Dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(vocabulary, outfile)
    np.save(emb_file, emb)
    print("# TrP all done.")

    train_file = "./dataset/tep/data/train.json" 
    dev_file = "./dataset/tep/data/dev.json"
    test_file = "./dataset/tep/data/test.json"

    vocab_file = "./dataset/tep/vocab/vocab.pkl"
    emb_file = "./dataset/tep/vocab/embedding.npy"

    print("# Loading TeP files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)

    v = build_vocab(train_tokens, vocab, 0)

    print("# Calculating TeP oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
    print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))
    
    print("# Dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(vocabulary, outfile)
    np.save(emb_file, emb)
    print("# TeP all done.")

    train_file = "./dataset/pp/data/train.json" 
    dev_file = "./dataset/pp/data/dev.json"
    test_file = "./dataset/pp/data/test.json"

    vocab_file = "./dataset/pp/vocab/vocab.pkl"
    emb_file = "./dataset/pp/vocab/embedding.npy"

    print("# Loading PiP files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)

    v = build_vocab(train_tokens, vocab, 0)

    print("# Calculating PiP oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
    print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))
    
    print("# Dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(vocabulary, outfile)
    np.save(emb_file, emb)
    print("# PiP all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['token']
            ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
            # do not create vocab for entity words
            ts[ss:se+1] = ['<PAD>']*(se-ss+1)
            ts[os:oe+1] = ['<PAD>']*(oe-os+1)
            tokens += list(filter(lambda t: t!='<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    PAD_TOKEN = '<PAD>'
    PAD_ID = 0
    UNK_TOKEN = '<UNK>'
    UNK_ID = 1
    VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]
    v = VOCAB_PREFIX + entity_masks() + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """ Get all entity mask tokens as a list. """
    PAD_TOKEN = '<PAD>'
    PAD_ID = 0
    UNK_TOKEN = '<UNK>'
    UNK_ID = 1
    masks = []
    SUBJ_NER_TO_ID = {
        PAD_TOKEN: 0, 
        UNK_TOKEN: 1,
        'treatment': 2,
        'problem': 3,
        'test': 4
    }
    OBJ_NER_TO_ID = {
        PAD_TOKEN: 0, 
        UNK_TOKEN: 1,
        'treatment': 2,
        'problem': 3,
        'test': 4
    }
    subj_entities = list(SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

if __name__ == '__main__':
    main()
