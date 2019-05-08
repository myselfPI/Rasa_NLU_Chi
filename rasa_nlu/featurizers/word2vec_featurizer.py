from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import typing
from typing import Any
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    from gensim.models import FastText
    from builtins import str


class Word2VecFeaturizer(Featurizer):
    name = "intent_featurizer_word2vec"

    provides = ["text_features"]

    requires = ["tokens", "word2vec_feature_extractor"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim", "numpy"]

    def ndim(self, feature_extractor):
        # type: (FastText.vector_size) -> int

        return feature_extractor.vector_size

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        word2vec_feature_extractor = self._word2vec_feature_extractor(**kwargs)
        for example in training_data.intent_examples:
            print("tokens")
            features = self.features_for_tokens(example.get("tokens"),
                                                word2vec_feature_extractor)
            print("features:")
            print(features)
            example.set("text_features",
                        self._combine_with_existing_text_features(
                                example, features))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        word2vec_feature_extractor = self._word2vec_feature_extractor(**kwargs)
        features = self.features_for_tokens(message.get("tokens"),
                                            word2vec_feature_extractor)
        message.set("text_features",
                    self._combine_with_existing_text_features(message,
                                                              features))

    def _word2vec_feature_extractor(self, **kwargs):
        word2vec_feature_extractor = kwargs.get("word2vec_feature_extractor")
        if not word2vec_feature_extractor:
            raise Exception("Failed to train 'intent_featurizer_word2vec'. "
                            "Missing a proper gensim.model.FastText feature extractor. "
                            "Make sure this component is preceded by "
                            "the 'nlp_word2vec' component in the pipeline "
                            "configuration.")
        return word2vec_feature_extractor

    def features_for_tokens(self, tokens, feature_extractor):
        # type: (List[Token], mitie.total_word_feature_extractor) -> np.ndarray

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            print(token.text)
            vec += feature_extractor[token.text]
        if tokens:
            return vec / len(tokens)
        else:
            return vec
