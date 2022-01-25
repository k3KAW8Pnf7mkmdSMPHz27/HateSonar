"""
Model API.
"""

import importlib.resources

import numpy as np
import onnxruntime as rt


class Sonar:
    _map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}

    def __init__(self):
        with importlib.resources.path("hatesonar.data", "pipeline.onnx") as pipeline_file:
            self.pipeline = rt.InferenceSession(str(pipeline_file.absolute()))

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
