#!/usr/bin/env python
import pandas as pd
import numpy as np

def load_brofile(filename, fields_to_use):
    fields = ['ts',
              'uid',
              'orig_h',
              'orig_p',
              'resp_h',
              'resp_p',
              'trans_depth',
              'method',
              'host',
              'uri',
              'referrer',
              'user_agent',
              'request_body_len',
              'response_body_len',
              'status_code',
              'status_msg',
              'info_code',
              'info_msg',
              'filename',
              'tags',
              'username',
              'password',
              'proxied orig_fuids',
              'orig_mime_types',
              'resp_fuids',
              'resp_mime_types']

    df = pd.read_csv(filename,
                     header=None,
                     sep='\t',
                     names=fields,
                     skiprows=8,
                     skipfooter=1,
                     index_col=False,
                     quotechar=None,
                     quoting=3,
                     engine='python')

    return df[fields_to_use]

def create_noise_contrast(df, num_samples):
    """
      Create a noise contrasted dataframe from a dataframe. We do this
      by sampling columns with replacement one at a time from the original
      data, and then stitching those columns together into de-correlated rows.

      Parameters
      ----------
      df : dataframe
          The enhanced HTTP log dataframe
      num_samples : int
            Number of new rows to create

      Returns
      -------
      newDf : dataframe

    """

    newDf = None
    for field in list(df):
        #sample the column with replacement.
        df1 = df[[field]].sample(n=num_samples, replace=True).reset_index(drop=True)

        #add the new column to the answer (or start a new df if this is the first column)
        if (newDf is not None):
            newDf = pd.concat([newDf, df1], axis = 1)
        else:
            newDf = df1


    return newDf
