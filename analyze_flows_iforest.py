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
    usage = """analyze_flows [options] inputfile"""
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
                      default=50, help="number of trees in random forest")

    Start=time.time()
    (opts, args) = parser.parse_args()

    if len(args)!=1:
        parser.error('Incorrect number of arguments')

    #load the http data in to a data frame
    print('Loading HTTP data')
    df = load_brofile(args[0], fields_to_use)

    print('Loading trained model')
    #read iForest data
    clf = joblib.load(opts.iforestfile)
    vectorizers = joblib.load(opts.vectorizerfile)


    total_rows = len(df.index)
    if opts.verbose: print('Total number of rows: %d' % total_rows)


    enhancedDf = enhance_flow(df)
    #construct some vectorizers based on the data in the DF. We need to vectorize future log files the exact same way so we
    # will be saving these vectorizers to a file.

    #vectorizers = build_vectorizers(enhancedDf, max_features=opts.maxfeaturesperbag, ngram_size=opts.ngramsize, verbose=opts.verbose)

    # use the vectorizers to featureize our DF into a numeric feature dataframe
    featureMatrix = featureize(enhancedDf, vectorizers, verbose=opts.verbose)


    print('Calculating features')
    #get a numberic feature dataframe using our flow enhancer and featurizer

    #clf = IsolationForest(n_jobs=4, n_estimators=opts.numtrees, oob_score=True)



    featureMatrix['prediction']= clf.decision_function(featureMatrix)

    print
    print('Analyzing')
    #get the class-1 (outlier/anomaly) rows from the feature matrix, and drop the prediction so we can investigate them

    ##From Here
    Left=0.001 
    Right=0.01
    
    k=len(featureMatrix['prediction'])/2

    Label= [0]*k + [1]*k

    print(len(Label))
    print(len(featureMatrix['prediction']))
    fpr, tpr, thresholds = roc_curve(Label, featureMatrix['prediction'])
    
    F=interpolate.interp1d(fpr, tpr)
    x=np.logspace(np.log10(Left), np.log10(Right))
    y=F(x)
    roc_auc=auc(x, y)
    print '%.6f'%(roc_auc)

    plt.figure()
    plt.xscale('log')
    plt.plot(fpr, tpr, ls='None', color='red')
    plt.plot(x,y, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')


    plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
    plt.plot(fpr, tpr)
    plt.savefig("fig3.png")
    plt.show()
    
    

    

    

    outliers = featureMatrix[featureMatrix.prediction <= 0.1].drop('prediction', axis=1)
    num_outliers = len(outliers.index)
    print 'detected %d anomalies out of %d total rows (%.2f%%)' % (num_outliers, total_rows, (num_outliers * 1.0 / total_rows)*100)

    Min, Sec= divmod( int(time.time() - Start), 60 )
    print Min, Sec

    target= open('Results', 'a')
    target.write(str(Min)+' ')
    target.write(str(Sec)+' ')
    target.write(str(roc_auc))
    target.write("\n")
    target.close()



    # if (opts.verbose):
    #     print 'investigating all the outliers'
    #     #investigate each outlier (determine the most influential columns in the prediction)
    #     prediction, bias, contributions = ti.predict(clf, outliers)
    #     print 'done'
    #     print(contributions.shape)
    ##To Here

    #i=0
    #for each anomaly
    #for index, row in outliers.iterrows():
        #print('-----------------------------------------')
        #print 'line ',index
        #find the row in the original data of the anomaly. print it out as CSV.
        #print pd.DataFrame(df.iloc[index]).T.to_csv(header=False, index=False)
        #if (opts.verbose):
            #if we are verbose print out the investigation by zipping the heavily weighted columns with the appropriate features
         #   instancecontributions = zip(contributions[i], outliers.columns.values)
            #print "Top feature contributions to class 1:"
            #for (c, feature) in sorted(instancecontributions, key=lambda (c,f): c[1], reverse=True)[:10]:
             # print '  ',feature, c[1]
        #i=i+1
