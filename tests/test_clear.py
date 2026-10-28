from pk_model.clear import linear

def test_clear_func():
    
    linear_clearance = linear(cl=20)
    
    assert linear_clearance(1) == 20
    assert linear_clearance(2) == 20 
    assert linear_clearance(20) == 20 