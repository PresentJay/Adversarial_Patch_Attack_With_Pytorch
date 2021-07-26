"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import pickle
import traceback
from modules.initializer import initializer
from modules.patch_generator import patch_genetator
from modules.patch_validator import patch_validator


if __name__ == '__main__':
    classifier, dataset, args, log = initializer()
    patch = patch_genetator(classifier, dataset, args, log)
    patch_validator(patch, classifier, dataset, args, log)