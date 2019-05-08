from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import traceback
import typing
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata

if typing.TYPE_CHECKING:
    from gensim.models import FastText


# from gensim.models import FastText
# model_ted = FastText(sentences_ted, size=100, window=5, min_count=5, workers=4,sg=1)
model = FastText.load_fasttext_format('/Users/kaiguo/Downloads/cc.zh.300.bin')
print(model.vector_size)
oov_vector = model['你好']
# print(oov_vector)

class Word2VecNLP(Component):
    name = "nlp_word2vec"

    provides = ["word2vec_feature_extractor","word2vec_file"]

    defaults = {
        # name of the language model to load - this contains
        # the MITIE feature extractor
        #"model": os.path.join("data", "total_word_feature_extractor.dat"),
        "model": '/Users/kaiguo/Downloads/cc.zh.300.bin'
    }

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 extractor=None
                 ):
        # type: (...) -> None
        """Construct a new language model from the MITIE framework."""

        super(Word2VecNLP, self).__init__(component_config)

        self.extractor = extractor

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim"]

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> MitieNLP
        from gensim.models import FastText
        from gensim.test.utils import get_tmpfile
        traceback.print_stack()
        component_conf = cfg.for_component(cls.name, cls.defaults)
        model_file = component_conf.get("model")
        if not model_file:
            raise Exception("The Word2Vec component 'nlp_word2vec' needs "
                            "the configuration value for 'model'."
                            "Please take a look at the "
                            "documentation in the pipeline section "
                            "to get more info about this "
                            "parameter.")
        fname = get_tmpfile("/Users/kaiguo/Downloads/cc.zh.300.new")
        extractor = FastText.load(fname)
        # extractor = FastText.load_fasttext_format('/Users/kaiguo/Downloads/cc.zh.300.bin')
        cls.ensure_proper_language_model(extractor)

        return Word2VecNLP(component_conf, extractor)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        component_meta = model_metadata.for_component(cls.name)

        word2vec_file = component_meta.get("model", None)
        if word2vec_file is not None:
            return cls.name + "-" + str(os.path.abspath(word2vec_file))
        else:
            return None

    def provide_context(self):
        # type: () -> Dict[Text, Any]
        traceback.print_stack()
        return {"word2vec_feature_extractor": self.extractor,
                "word2vec_file": self.component_config.get("model")}

    @staticmethod
    def ensure_proper_language_model(extractor):
        # type: (Optional[mitie.total_word_feature_extractor]) -> None

        if extractor is None:
            raise Exception("Failed to load Word2Vec feature extractor. "
                            "Loading the model returned 'None'.")

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Word2VecNLP]
             **kwargs  # type: **Anything
             ):
        # type: (...) -> Word2VecNLP
        # import mitie
        from gensim.test.utils import get_tmpfile
        from gensim.models import FastText
        if cached_component:
            return cached_component

        component_meta = model_metadata.for_component(cls.name)
        word2vec_file = component_meta.get("model")
        # extractor = FastText.load_fasttext_format('/Users/kaiguo/Downloads/cc.zh.300.bin')
        fname = get_tmpfile("/Users/kaiguo/Downloads/cc.zh.300.new")
        extractor = FastText.load(fname)
        return cls(component_meta,
                   extractor)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "word2vec_feature_extractor_fingerprint": "1234567890",
            "model": self.component_config.get("model")
        }


