from roman_table import *

def test_is_detectable():
    test_seps = [4,   5,   6,   0.5, 50]
    test_fcs =  [1e-4,1e-5,1e-6,1e-1,1e-1]
    
    test_concurve = np.array([
        [1,5,   10,  15,  20,  25,  30,  40], # sep_mas
        [1,1e-5,1e-6,1e-7,1e-7,1e-7,1e-7,1] # contrast limit
    ])

    expected_out = np.array([ True,  True, False, False, False])

    assert np.all(expected_out == is_detectable(test_seps,test_fcs,test_concurve))


if __name__=='__main__':
    test_is_detectable()