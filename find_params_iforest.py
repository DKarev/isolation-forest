#!/usr/bin/env python

import pandas as pd
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
from optparse import OptionParser

from featureizer import featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile
import logging

from bisect import bisect
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

from sys import argv
from scipy import interpolate

import time

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from optparse import OptionParser

from featureizer import build_vectorizers,featureize
from clearcut_utils import load_brofile, create_noise_contrast

all_fields = ['resp_p',
              'method',
              'host',
              'uri',
              'referrer',
              'user_agent',
              'request_body_len',
              'response_body_len',
              'status_code',

              'browser_string',
              'URIparams', 'URItokens',
              'subdomain', 'tld',
              'domainNameLength', 'domainNameDots',
              'uriSlashes', 'userAgentLength',
              'referrerPresent', 'numURIParams']

fields_to_use = ['resp_p',
                 'method',
                 'host',
                 'uri',
                 'referrer',
                 'user_agent',
                 'request_body_len',
                 'response_body_len',
                 'status_code']
logging.basicConfig()




def Eval(clargs):    
    __version__ = '1.0'
    usage = """train_flows [options] normaldatafile"""
    parser = OptionParser(usage=usage, version=__version__)

    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="")
    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")
    parser.add_option("-o", "--maliciousdatafile", action="store", type="string", \
                      default=None, help="An optional file of malicious http logs")
    parser.add_option("-m", "--maxfeaturesperbag", action="store", type="int", \
                      default=100, help="maximum number of features per bag")
    parser.add_option("-g", "--ngramsize", action="store", type="int", \
                      default=7, help="ngram size")

    parser.add_option("-f", "--features", action="store", type="string", \
                      default="01000100111111111111", help="An optional file for choosing which features to be extracted")
    parser.add_option("-t", "--maxtrainingfeatures", action="store", type="int", \
                      default=50000, help="maximum number of rows to train with per class")
    parser.add_option("-n", "--numtrees", action="store", type="int", \
                      default=200, help="number of trees in isolation forest")
    parser.add_option("-s", "--numsamples", action="store", type="int", \
                      default=8192, help="number of samples in each tree")


    Start=time.time()
    (opts, args) = parser.parse_args(clargs)

    if len(args) != 2:
        parser.error('Incorrect number of arguments')

    ftu=[]
    features = opts.features

    for i, j in enumerate(features):
      if opts.verbose: print(j, all_fields[i])
      if j == 1 or j=='1':
        ftu.append(all_fields[i])

    if opts.verbose: print ftu
    #ftu = ['method', 'user_agent', 'status_code']


    # load the http data in to a data frame
    print('Loading HTTP data')
    df = load_brofile(args[0], fields_to_use)
    trainDf = load_brofile(args[1], fields_to_use)


    total_rows = len(df.index)
    if opts.verbose: print('Total number of rows: %d' % total_rows)
    if opts.maliciousdatafile != None:
      print('Reading malicious training data')
      df1 = load_brofile(opts.maliciousdatafile, fields_to_use)
      if opts.verbose: print('Read malicious data with %s rows ' % len(df1.index))
      #if (len(df1.index) > opts.maxtrainingfeatures):
      #  if opts.verbose: print('Too many malicious samples for training, downsampling to %d' % opts.maxtrainingfeatures)
      #  df1 = df1.sample(n=opts.maxtrainingfeatures)

      #set the classes of the dataframes and then stitch them together in to one big dataframe
      df['class'] = 0
      df1['class'] = 1
      classedDf = pd.concat([df,df1], ignore_index=True)
    else:
      #we weren't passed a file containing class-1 data, so we should generate some of our own.
      noiseDf = create_noise_contrast(df, numSamples)
      if opts.verbose: print('Added %s rows of generated malicious data'%numSamples)
      df['class'] = 0
      noiseDf['class'] = 1
      classedDf = pd.concat([df,noiseDf], ignore_index=True)

    #that doesn't matter
    trainDf['class']=0;


    #spliting into training and evaluation sets 
    classedDf['is_train']=False
    trainDf['is_train']=True

    enhancedDf = enhance_flow(pd.concat([trainDf,classedDf], ignore_index=True), ftu)
    # construct some vectorizers based on the data in the DF. We need to vectorize future log files the exact same way so we
    # will be saving these vectorizers to a file.

    vectorizers = build_vectorizers(enhancedDf, ftu, max_features=opts.maxfeaturesperbag, ngram_size=opts.ngramsize, verbose=opts.verbose)

    #use the vectorizers to featureize our DF into a numeric feature dataframe
    featureMatrix = featureize(enhancedDf, ftu, vectorizers, verbose=opts.verbose)

    #add the class column back in (it wasn't featurized by itself)
    featureMatrix['class'] = enhancedDf['class']
    featureMatrix['is_train'] = enhancedDf['is_train']


    #split out the train and test df's into separate objects
    train, test = featureMatrix[featureMatrix['is_train']==True], featureMatrix[featureMatrix['is_train']==False]

    #drop the is_train column, we don't need it anymore
    train = train.drop('is_train', axis=1)
    test = test.drop('is_train', axis=1)


    #print('Calculating features')


    Trees=opts.numtrees
    Samples=opts.numsamples
    clf = IsolationForest(n_estimators=Trees, max_samples=Samples)

    
    clf.fit(train.drop('class', axis=1))

    testnoclass = test.drop('class', axis=1)

    print('Predicting')

    test.is_copy = False

    test['prediction'] = clf.decision_function(testnoclass) + 0.5

    print('Analyzing')
    #get the class-1 (outlier/anomaly) rows from the feature matrix, and drop the prediction so we can investigate them

    ##From Here
    Left=0.001 
    Right=0.01
    
    fpr, tpr, thresholds = roc_curve(test['class'], test['prediction'], pos_label=0)
    
    F=interpolate.interp1d(fpr, tpr, assume_sorted=True)
    x=np.logspace(np.log10(Left), np.log10(Right))
    y=F(x)
    roc_auc=auc(x, y)

    plt.figure()
    plt.xscale('log')

    plt.plot(fpr, tpr, color='b')
    plt.plot(x,y, color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')


    plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
    plt.savefig("fig3.png")
    plt.show()

    print('Area Under the Curve = %.6f' %(roc_auc))



    Min, Sec= divmod( int(time.time() - Start), 60 )
    #print Min, Sec

    target= open('Results.txt', 'a')
    target.write(str(Trees)+' ')
    target.write(str(Samples)+' ')
    target.write(str(Min)+' ')
    target.write(str(Sec)+' ')
    target.write(str(roc_auc))
    target.write("\n")
    target.write(str(features))
    target.close()

    
    print("Minutes: %d, Seconds: %d" % (int(Min), int(Sec)) )
    return roc_auc 

    #joblib.dump(vectorizers, opts.vectorizerfile)
    #joblib.dump(clf, opts.iforestfile)

if __name__ == "__main__":
    import sys
    Eval(sys.argv[1:])
