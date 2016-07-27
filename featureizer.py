#!/usr/bin/env python
import pandas as pd
import math, string, sys
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from urlparse import urlparse, parse_qs
import logging

logging.basicConfig()


def build_vectorizers(df, ftu, max_features=100, ngram_size=7, verbose=False):
    """
      Build some vectorizers based on an enhanced http dataframe

      Parameters
      ----------
      df : dataframe
          The enhanced HTTP log dataframe
      max_features : int, optional
          the most features per bag to use. Default 100
      ngram_size : int, optional
          The size of ngrams to use for bag of ngrams. Defaults to 7
      verbose: boolean, optional
          Controls Verbosity level
      ftu: fields to use
          Controls which fields to be used.

      Returns
      -------
      vectorizers : {String -> TfidfVectorizer}
            A map of feature -> vectorizer for use in the featureize function.

    """
    print('\nBuilding Vectorizers')
    vectorizers = {}
    #create bag of ngram vectorizers
    for feature in ['user_agent','uri','referrer','host', 'subdomain']:
      if feature in ftu:
        if verbose: print('Creating BON Vectorizer for %s' % feature)
        vectorizer = TfidfVectorizer(analyzer='char',max_features = max_features,ngram_range=(ngram_size,ngram_size))
        vectorizers[feature] = vectorizer.fit(df[feature].astype(str))

    #create bag of words vectorizers
    for feature in ['method','status_code','resp_p', 'URIparams', 'browser_string', 'tld']:
    #for feature in ['method','status_code']:
      if feature in ftu:
        if verbose: print('Creating BOW Vectorizer for %s' % feature)
        vectorizer = TfidfVectorizer(analyzer='word',max_features = max_features)
        vectorizers[feature] = vectorizer.fit(df[feature].astype(str))

    return vectorizers

def featureize(df, ftu, vectorizers, verbose=False):
    # type: (object, object, object) -> object
    """
      Featurize an enhanced http dataframe

      Parameters
      ----------
      df : dataframe
          The enhanced HTTP log dataframe
      vectorizers : {String -> TfidfVectorizer}
            A map of feature -> vectorizer
      verbose: boolean, optional
          Controls Verbosity level

      Returns
      -------
      featureMatrix : numeric dataframe
            A featurized dataframe

    """
    if verbose: print('\nExtracting features')
    
    bow_features = []
    #featurize using the vectorizers.
    
    
    for feature in ['user_agent','uri','referrer','host', 'subdomain', 'method','status_code','resp_p', 'URIparams', 'browser_string', 'tld']:
    #for feature in ['user_agent', 'method', 'status_code']:
      if feature in ftu:
        if verbose: print('Featurizing %s' % feature)
        single_feature_matrix = vectorizers[feature].transform(df[feature].astype(str))
        if verbose: print('  Dim of %s: %s' % (feature,single_feature_matrix.shape[1]))
        single_df = DataFrame(single_feature_matrix.toarray())
        single_df.rename(columns=lambda x: feature+"."+vectorizers[feature].get_feature_names()[x], inplace=True)
        bow_features.append(single_df)

    featureMatrix = pd.concat(bow_features, axis=1)
    
    #add some other numeric features that are functions of columns
    if 'domainNameLength' in ftu:
      featureMatrix['domainNameLength'] = df['host'].apply(len)
    if 'domainNameDots' in ftu:
      featureMatrix['domainNameDots'] = df['host'].apply(lambda dn: dn.count('.'))
    if 'uriSlashes' in ftu:
      featureMatrix['uriSlashes'] = df['uri'].apply(lambda dn: dn.count('/'))

    if 'userAgentLength' in ftu:
      featureMatrix['userAgentLength'] = df['user_agent'].apply(len)
    if 'userAgentEntropy' in ftu:
      featureMatrix['userAgentEntropy'] = df['user_agent'].apply(H)

    if 'subdomain' in ftu:
      featureMatrix['subdomainEntropy'] = df['subdomain'].apply(H)
    if 'request_body_len' in ftu:
      featureMatrix['request_body_len'] = df['request_body_len']
    if 'response_body_len' in ftu:
      featureMatrix['response_body_len'] = df['response_body_len']

    if 'referrerPresent' in ftu:
      featureMatrix['referrerPresent'] = df['referrer'].apply(lambda r: 0.0 if (r=='-') else 1.0)
    
    def countParams(uri):
        fullUri = 'http://bogus.com/'+uri
        parseResult = parse_qs(urlparse(fullUri).query)
        return len(parseResult)
    
    if 'numURIParams' in ftu:
      featureMatrix['numURIParams'] = df['uri'].apply(countParams)
      
    if 'URIparams' in ftu:
      featureMatrix['URIParamsKeyEntropy'] = df['URIparams'].apply(H)
    if 'URItokens' in ftu:
      featureMatrix['URIParamsTokensEntropy'] = df['URItokens'].apply(H)

    if verbose: print('Feature matrix generated with %s columns' % featureMatrix.shape[1])

    return featureMatrix


# Some functions to compute entropy of strings
#
# Borrowed from Ero Carrera
# http://blog.dkbza.org/2007/05/scanning-data-for-entropy-anomalies.html
def range_bytes (): return range(256)
def range_printable(): return (ord(c) for c in string.printable)
def H(data, iterator=range_bytes):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy


