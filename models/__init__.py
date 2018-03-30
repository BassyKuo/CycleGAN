import sys, inspect
from models.cyclegan import CYCLEGAN
from models.cyclegan_photo2label import CYCLEGAN_PHOTO2LABEL as PHOTO2LABEL


__list = [model[0] for model in inspect.getmembers(sys.modules[__name__], inspect.isclass)]
