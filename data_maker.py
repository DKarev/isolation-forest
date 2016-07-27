#!/usr/bin/env python

import pandas
from sklearn.externals import joblib
from treeinterpreter import treeinterpreter as ti
from optparse import OptionParser


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from featureizer import featureize
from flowenhancer import enhance_flow
from clearcut_utils import load_brofile
from train_flows_rf import fields_to_use
import logging
import json
import numpy as np


from featureizer import build_vectorizers,featureize
from clearcut_utils import load_brofile, create_noise_contrast

logging.basicConfig()

if __name__ == "__main__":
    __version__ = '1.0'
    usage = """data_maker [options] normaldatafile"""
    parser = OptionParser(usage=usage, version=__version__)

    (opts, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Incorrect number of arguments')

    with open(args[0]) as f:
      #w, h = [int(x) for x in next(f).split()] # read first line
      trees = []
      samples = []
      time = []
      auc = []
      array = []
      for line in f:
          x, y, t, a = [float(x) for x in line.split()] 
          trees.append(x)
          samples.append(y)
          time.append(t)
          auc.append(a)

    #print(trees)
    #print(samples)
    #print(time)
    #print(auc)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(np.log2(samples), trees,  time, color='black')
    ax.set_xlabel('$Log_2$ $of$ $number$ $of$ $nodes$ $in$ $the$ $tree$')
    ax.set_ylabel('$Number$ $of$ $trees$')
    ax.set_zlabel('$Time$ $in$ $seconds$')
    plt.show()
    plt.savefig('3DFigP.png')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot( trees, np.log2(samples),  auc, color='black')
    ax.set_ylabel('$Log_2$ $of$ $number$ $of$ $nodes$ $in$ $the$ $tree$')
    ax.set_xlabel('$Number$ $of$ $trees$')
    ax.set_zlabel('$Area$ $under$ $the$ $curve$')
    plt.show()
    plt.savefig('3DFigP1.png')



    #total_df = pd.concat([df,df1], ignore_index=True)

    #f.close()

    #enhancedDf = enhance_flow(pd.concat([trainDf,classedDf], ignore_index=True))


    #joblib.dump(vectorizers, opts.vectorizerfile)
    #joblib.dump(clf, opts.iforestfile)