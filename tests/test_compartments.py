import pytest
import numpy as np

from pk_model.compartments import Comp, Peripheral

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
    
def test_bad_peripheral_params():

    with pytest.raises(ValueError):
        peripheral = Peripheral(-1, 20)
    
    with pytest.raises(ValueError):
        peripheral = Peripheral(10,-2)
        
def test_compartments_container(comp_three_p):
    
    assert comp_three_p.central_volume == 15
    assert len(comp_three_p.peripherals) == 3
    
def test_get_volumes(comp_three_p):
    
    np.testing.assert_array_equal(comp_three_p.volumes, np.array([15,1,3,5]))        

def test_no_peripherals():
    
    comp = Comp(central_volume=15)
    
    np.testing.assert_array_equal(comp.volumes, np.array([15]))     #Only central volume.
    np.testing.assert_array_equal(comp.peripherals, np.array([]))   #No peripherals.
    np.testing.assert_array_equal(comp.Q, np.array([]))             #(hence) no transition rates.