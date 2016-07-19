#!/usr/bin/env python

import pandas as pd
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
from optparse import OptionParser

from featureizer import featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile
from train_flows_rf import fields_to_use
import logging

from sys import argv
import numpy as np
from sklearn.ensemble import IsolationForest
from optparse import OptionParser

import time

from featureizer import build_vectorizers, featureize
from clearcut_utils import load_brofile, create_noise_contrast

logging.basicConfig()

fields_to_use = ['uid', 'resp_p',
                 'method',
                 'host',
                 'uri',
                 'referrer',
                 'user_agent',
                 'request_body_len',
                 'response_body_len',
                 'status_code']

if __name__ == "__main__":
    __version__ = '1.0'
    usage = """train_flows [options] normaldatafile"""
    parser = OptionParser(usage=usage, version=__version__)
    parser.add_option("-f", "--iforestfile", action="store", type="string", \
                      default='/tmp/rf.pkl', help="")
    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="")
    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")
    usage = """train_flows [options] normaldatafile"""
    parser.add_option("-o", "--maliciousdatafile", action="store", type="string", \
                      default=None, help="An optional file of malicious http logs")
    parser.add_option("-m", "--maxfeaturesperbag", action="store", type="int", \
                      default=100, help="maximum number of features per bag")
    parser.add_option("-g", "--ngramsize", action="store", type="int", \
                      default=7, help="ngram size")

    parser.add_option("-t", "--maxtrainingfeatures", action="store", type="int", \
                      default=50000, help="maximum number of rows to train with per class")
    parser.add_option("-n", "--numtrees", action="store", type="int", \
                      default=100, help="number of trees in isolation forest")
    parser.add_option("-s", "--numsamples", action="store", type="int", \
                      default=256, help="number of samples in each tree")

    Start=time.time()
    (opts, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')

    # load the http data in to a data frame
    print('Loading HTTP data')
    df = load_brofile(args[0], fields_to_use)

    total_rows = len(df.index)
    if opts.verbose: print('Total number of rows: %d' % total_rows)

    enhancedDf = enhance_flow(df)
    # construct some vectorizers based on the data in the DF. We need to vectorize future log files the exact same way so we
    # will be saving these vectorizers to a file.

    vectorizers = build_vectorizers(enhancedDf, max_features=opts.maxfeaturesperbag, ngram_size=opts.ngramsize,
                                    verbose=opts.verbose)

    # use the vectorizers to featureize our DF into a numeric feature dataframe
    featureMatrix = featureize(enhancedDf, vectorizers, verbose=opts.verbose)

    print('Calculating features')
    # get a numberic feature dataframe using our flow enhancer and featurizer

    # clf = IsolationForest(n_jobs=4, n_estimators=opts.numtrees, oob_score=True)

    Trees=opts.numtrees
    Samples=opts.numsamples
    clf = IsolationForest(n_estimators=Trees, max_samples=Samples)

    print('Calculating features2')

    # predict the class of each row using the isolation forest
    clf.fit(featureMatrix)

    Min, Sec= divmod( int(time.time() - Start), 60 )
    print Min, Sec

    target= open('Results', 'a')
    target.write(str(Trees)+' ')
    target.write(str(Samples)+' ')
    target.write(str(Min)+' ')
    target.write(str(Sec)+' ')
    target.close()

    print("\a")
    print("\a")

    #print("Minutes: %d, Seconds: %d" % (int(Min), int(Sec)) ) 

    joblib.dump(vectorizers, opts.vectorizerfile)
    joblib.dump(clf, opts.iforestfile)