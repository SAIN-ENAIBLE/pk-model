from pk_model.compartments import Comp, Peripheral
import pytest

def test_good_peripheral_params():
    
    peripheral = Peripheral(10,20)
    
    assert peripheral.volume == 10
    assert peripheral.transition_rate == 20
    
def test_bad_peripheral_params():

    with pytest.raises(ValueError):
        peripheral = Peripheral(-1, 20)
    
    with pytest.raises(ValueError):
        peripheral = Peripheral(10,-2)
        
def test_compartments_container():
    
    peripherals = (Peripheral(1,2),
                   Peripheral(3,4),
                   Peripheral(5,6))
    
    comp = Comp(central_volume=15,
                peripherals=peripherals)
    
    assert comp.central_volume == 15
    assert len(comp.peripherals) == 3
    