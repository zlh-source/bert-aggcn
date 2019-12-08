#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import sklearn.metrics
import numpy

NO_RELATION = ('NoRE_PIP','NoRE_TeP', 'NoRE_TrP')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=False):
    cm = sklearn.metrics.confusion_matrix(key, prediction)
    nc, nc2 = cm.shape
    assert nc==nc2
    pres = numpy.zeros(nc); recs = numpy.zeros(nc); f1s = numpy.zeros(nc)
    print("key {}".format(key))
    print("predictions {}".format(prediction))
    print(cm)
    tp_a = 0; fn_a = 0; fp_a = 0
    labels = list(set(key+prediction))
    ncstart = 0
    for item in NO_RELATION:
        if item in labels:
            ncstart = 1
            break
    for c in range(ncstart,nc):
        tp = cm[c,c]; tp_a += tp
        mask = numpy.ones(nc,dtype=bool)
        mask[c] = 0 
        fn = numpy.sum( cm[c, mask] ); fn_a += fn
        fp = numpy.sum( cm[mask, c] ); fp_a += fp
        if tp+fp == 0:
            pre = 1
        else:
            pre = tp / (tp+fp)
        if tp+fn == 0:
            rec = 1
        else:
            rec = tp / (tp+fn)
        if pre+rec == 0:
            f = 0
        else:
            f = 2*pre*rec / (pre+rec)
        pres[c] = pre; recs[c] = rec; f1s[c] = f
    if tp_a+fp_a == 0:
        mipre = 1
    else:
        mipre = tp_a / (tp_a+fp_a)
    if tp_a+fn_a == 0:
        mirec = 1
    else:
        mirec = tp_a / (tp_a+fn_a)
    if mipre+mirec == 0:
        mif = 0
    else:
        mif = 2*mipre*mirec / (mipre+mirec)
    
    print( "Precision (micro): {:.3%}".format(mipre) )
    print( "   Recall (micro): {:.3%}".format(mirec) )
    print( "       F1 (micro): {:.3%}".format(mif) )
    return (mipre, mirec, mif)

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

