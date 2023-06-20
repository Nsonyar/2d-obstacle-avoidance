
import sys
import pytest
from source.ss20_02_feature_extraction.feature_extraction import FeatureExtraction
'''
    1. Create your inputs
    2. Execute the code, capturing the output
    3. Compare the output with an expected result

    python2 -m pytest test_feature_extraction.py
'''

def test_dictionary():
    FE = FeatureExtraction()
    range_arr = FE.get_ranges(200,5,666)
    result = len(FE.get_dictionary(range_arr))
    assert result == 3

def test_string_int_exception():
    ''' Function checks if Exception is raised on all functions from the class
        FeatureExtraction if given a string or int. All functions except init
        and doc are supposed to raise an exception. The functions are first 
        stored in a string and then executed with eval(). The first 25 function
        which are python specific are not evaluated here.
    '''
    FE = FeatureExtraction()
    array = dir(FeatureExtraction)
    print(array)
    typeArray = ['\'stringTest\'',1]
    
    for x in range(len(typeArray)):
        print(x)
        array = ['FE.'+ i + '('+ str(typeArray[x]) +')' for i in array]
        for i in range(25,len(array)):
            print(array[i])
            with pytest.raises(Exception):
                eval(array[i])



