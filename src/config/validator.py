# TODO: apply Google style Python Docstring

""" 
define validation codes

using assert notation
"""

def NoneValidation(**kwargs):
    for key, val in kwargs.items():
        assert val is not None, f'you must define {key}! please read our doc'
    
