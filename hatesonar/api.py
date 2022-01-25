"""
Model API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import onnxruntime as rt


class Sonar(object):
    _map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}

    def __init__(self):
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        print(base_dir)
        print(os.listdir(base_dir))
        pipeline_file = os.path.join(base_dir, "pipeline.onnx")
        self.pipeline = rt.InferenceSession(pipeline_file)

    def ping(self, text: str) -> dict:
        assert isinstance(text, str)

        proba = self.pipeline.run(None, {"input": [text]})[1][0]

        res = {
            "text": text,
            "top_class": Sonar._map[np.argmax(proba)],
            "classes": [
                {"class_name": Sonar._map[k], "confidence": proba[k]}
                for k in sorted(Sonar._map)
            ],
        }

        return res
