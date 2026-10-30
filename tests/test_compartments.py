import pytest
import numpy as np

from pk_model.compartments import Comp, Peripheral

BAD = [0,-1,float('nan'),float('inf')]

@pytest.fixture
def comp_three_p():
    peripherals = (Peripheral(1,2),
                   Peripheral(3,4),
                   Peripheral(5,6))
        
    comp = Comp(central_volume=15,
                peripherals=peripherals)
        
    return comp

def test_good_peripheral_params():
    
    peripheral = Peripheral(10,20)
    
    assert peripheral.volume == 10
    assert peripheral.transition_rate == 20
    

@pytest.mark.parametrize(
    argnames="bad",
    argvalues=BAD
)
def test_peripheral_rejects_bad_volume(bad):
    with pytest.raises(ValueError, match="volume"):
        Peripheral(volume=bad, transition_rate=1)
    
@pytest.mark.parametrize(
    argnames="bad",
    argvalues=BAD
)
def test_peripheral_rejects_bad_rate(bad):
    with pytest.raises(ValueError, match="transition_rate"):
        Peripheral(volume=1, transition_rate=bad)

@pytest.mark.parametrize(
    argnames="bad",
    argvalues=BAD
)
def test_central_rejects_bad_volume(bad):
    with pytest.raises(ValueError, match="central_volume"):
        Comp(central_volume=bad)
        
def test_compartments_container(comp_three_p):
    
    assert comp_three_p.central_volume == 15
    assert len(comp_three_p.peripherals) == 3
    
def test_get_volumes(comp_three_p):
    
    np.testing.assert_array_equal(comp_three_p.volumes, np.array([15,1,3,5]))        

def test_get_transition_rates(comp_three_p):
    
    np.testing.assert_array_equal(comp_three_p.Q, np.array([2,4,6]))

def test_no_peripherals():
    
    comp = Comp(central_volume=15)
    
    np.testing.assert_array_equal(comp.volumes, np.array([15]))     #Only central volume.
    np.testing.assert_array_equal(comp.peripherals, np.array([]))   #No peripherals.
    np.testing.assert_array_equal(comp.Q, np.array([]))             #(hence) no transition rates.