from pk_model import dose
import pytest

def test_infusion_range():
    
    infusion_dose = dose.infusion(start=0,
                                  end=30,
                                  rate=15)
    
    assert infusion_dose(t=0) == 15
    assert infusion_dose(t=15.074) == 15
    assert infusion_dose(t=30.004) == 0

def test_negative_timestep():
    
    infusion_dose = dose.infusion(start=0,
                                      end=30,
                                      rate=15)
    
    with pytest.raises(ValueError):
            infusion_dose(t=-2)