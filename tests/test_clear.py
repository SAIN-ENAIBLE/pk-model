"""
Thoughts:
# How do we handle negative values here? 
- At construction, check CL, Vmax, or Km <=0 and reject
- *not* at call time, because ivp_solver rounding errors near zero can produce
concentrations which look like -1e12 ; one call would then stop the solver.

"""

import pytest

from pk_model import clear


def test_clear_func():
    
    constant_clearance = clear.constant(rate=20)
    
    assert constant_clearance(1,1) == 20
    assert constant_clearance(2,5) == 20 
    assert constant_clearance(20,20) == 20

def test_linear_clearance():
    
    linear_clearance = clear.linear(CL=20)
    
    assert linear_clearance(1, 1) == 20
    assert linear_clearance(1,2) == 40
    assert linear_clearance(1,2.205) == pytest.approx(44.1)
    
def test_michealis_menten():
    
    mm_clearance = clear.michaelis_menten(Vmax=10,
                                          Km=5)
    
    assert mm_clearance(1,0) == 0
    assert mm_clearance(1,5) == 5 # conc == Km -> Vmax/2
    
@pytest.mark.parametrize(argnames = "bad_cl",
                         argvalues= [-1, float("inf"), float("nan")])
def test_bad_CL_input(bad_cl):
    
    with pytest.raises(ValueError):
        clear.linear(CL=bad_cl)