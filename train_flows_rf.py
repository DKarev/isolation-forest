#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from optparse import OptionParser

from featureizer import build_vectorizers,featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile, create_noise_contrast
import logging
logging.basicConfig()

fields_to_use=['uid','resp_p',
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
    parser.add_option("-o", "--maliciousdatafile", action="store", type="string", \
                      default=None, help="An optional file of malicious http logs")
    parser.add_option("-f", "--randomforestfile", action="store", type="string", \
                      default='/tmp/rf.pkl', help="the location to store the forest classifier")
    parser.add_option("-x", "--vectorizerfile", action="store", type="string", \
                      default='/tmp/vectorizers.pkl', help="the location to store the vectorizer")

    parser.add_option("-m", "--maxfeaturesperbag", action="store", type="int", \
                          default=100, help="maximum number of features per bag")
    parser.add_option("-g", "--ngramsize", action="store", type="int", \
                      default=7, help="ngram size")

    parser.add_option("-t", "--maxtrainingfeatures", action="store", type="int", \
                      default=50000, help="maximum number of rows to train with per class")
    parser.add_option("-n", "--numtrees", action="store", type="int", \
                      default=50, help="number of trees in random forest")
    parser.add_option("-v", "--verbose", action="store_true", default=False, \
                      help="enable verbose output")

    (opts, args) = parser.parse_args()

    if len(args)!=1:
        parser.error('Incorrect number of arguments')

    print('Reading normal training data')
    df = load_brofile(args[0], fields_to_use)
    if opts.verbose: print('Read normal data with %s rows ' % len(df.index))

    numSamples = len(df.index)

    if (numSamples > opts.maxtrainingfeatures):
        if opts.verbose: print('Too many normal samples for training, downsampling to %d' % opts.maxtrainingfeatures)
        df = df.sample(n=opts.maxtrainingfeatures)
        numSamples = len(df.index)

    if opts.maliciousdatafile != None:
        print('Reading malicious training data')
        df1 = load_brofile(opts.maliciousdatafile, fields_to_use)
        if opts.verbose: print('Read malicious data with %s rows ' % len(df1.index))
        if (len(df1.index) > opts.maxtrainingfeatures):
            if opts.verbose: print('Too many malicious samples for training, downsampling to %d' % opts.maxtrainingfeatures)
            df1 = df1.sample(n=opts.maxtrainingfeatures)

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

    #add some useful columns to the data frame
    enhancedDf = enhance_flow(classedDf)

    if opts.verbose: print('Concatenated normal and malicious data, total of %s rows' % len(enhancedDf.index))

    #construct some vectorizers based on the data in the DF. We need to vectorize future log files the exact same way so we
    # will be saving these vectorizers to a file.
    vectorizers = build_vectorizers(enhancedDf, max_features=opts.maxfeaturesperbag, ngram_size=opts.ngramsize, verbose=opts.verbose)

    #use the vectorizers to featureize our DF into a numeric feature dataframe
    featureMatrix = featureize(enhancedDf, vectorizers, verbose=opts.verbose)

    #add the class column back in (it wasn't featurized by itself)
    featureMatrix['class'] = enhancedDf['class']

    #randomly assign 3/4 of the feature df to training and 1/4 to test
    featureMatrix['is_train'] = np.random.uniform(0, 1, len(featureMatrix)) <= .75

    #split out the train and test df's into separate objects
    train, test = featureMatrix[featureMatrix['is_train']==True], featureMatrix[featureMatrix['is_train']==False]

    #drop the is_train column, we don't need it anymore
    train = train.drop('is_train', axis=1)
    test = test.drop('is_train', axis=1)

    #create the random forest class and factorize the class column
    clf = RandomForestClassifier(n_jobs=4, n_estimators=opts.numtrees, oob_score=True)
    y, _ = pd.factorize(train['class'])

    #train the random forest on the training set, dropping the class column (since the trainer takes that as a separate argument)
    print('\nTraining')
    clf.fit(train.drop('class', axis=1), y)

    #remove the 'answers' from the test set
    testnoclass = test.drop('class', axis=1)


    #rank the features using some magic
    if opts.verbose:
        print("\nFeature ranking:")

        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        for f in range(testnoclass.shape[1]):
            if (importances[indices[f]] > 0.005):
                print("%d. feature %s (%f)" % (f + 1, testnoclass.columns.values[indices[f]], importances[indices[f]]))


    print('\nPredicting (class 0 is normal, class 1 is malicious)')

    #evaluate our results on the test set.
    test.is_copy = False
    test['prediction'] = clf.predict(testnoclass)
    print

    #group by class (the real answers) and prediction (what the RF said). we want these values to match for 'good' answers
    results=test.groupby(['class', 'prediction'])
    resultsagg = results.size()
    print(resultsagg)

    tp = float(resultsagg[1,1]) if (1,1) in resultsagg.index else 0
    fp = float(resultsagg[0,1]) if (0,1) in resultsagg.index else 0
    fn = float(resultsagg[1,0]) if (1,0) in resultsagg.index else 0
    f1 = 2*tp/(2*tp + fp + fn)
    print("F1 = %s" % f1)

    #save the vectorizers and trained RF file
    joblib.dump(vectorizers, opts.vectorizerfile)
    joblib.dump(clf, opts.randomforestfile)
