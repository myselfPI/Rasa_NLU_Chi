from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
import gensim

# import cloudpickle
# import io
# with io.open("cc.zh.300.bin.dat", 'wb') as f:
#     cloudpickle.dump(model, f)

# cap_path = datapath("cc.zh.300.bin")
# model = gensim.models.fasttext.load_facebook_vectors(cap_path)
# model = FastText.load_fasttext_format('/Users/kaiguo/Downloads/cc.zh.300.bin')

fname = get_tmpfile("/Users/kaiguo/Downloads/cc.zh.300.new")
# model.save(fname)
model = FastText.load(fname)
data = model.wv.__getitem__('jfoajfeoawfpawefjipfao')

# extractor = FastText.load(fname)

class ResetModel:

    def addition(x, y):
        added = x + y
        print(added)

    def subtraction(x, y):
        sub = x - y
        print(sub)

    def multiplication(x, y):
        mult = x * y
        print(mult)

    def division(x, y):
        div = x / y
        print(div)