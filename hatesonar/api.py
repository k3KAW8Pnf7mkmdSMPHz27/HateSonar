"""
Model API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import joblib
import numpy as np


class Sonar(object):
    _map = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}

    def __init__(self):
        BASE_DIR = os.path.join(os.path.dirname(__file__), './data')
        model_file = os.path.join(BASE_DIR, 'model.joblib')
        preprocessor_file = os.path.join(BASE_DIR, 'preprocess.joblib')
        self.estimator = joblib.load(model_file)
        self.preprocessor = joblib.load(preprocessor_file)

    def ping(self, text):
        assert isinstance(text, str)

        vector = self.preprocessor.transform([text])
        proba = self.estimator.predict_proba(vector)[0]
        mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}

        res = {
            'text': text,
            'top_class': Sonar._map[np.argmax(proba)],
            'classes': [
                {'class_name': Sonar._map[k],
                 'confidence': proba[k]}
                for k in sorted(Sonar._map)
            ]
        }

        return res
