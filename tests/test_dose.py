"""
Thoughts:
- Rather than multiple assertion lines e.g test_infusion_range, use pyest.mark.parametrise


"""

from pk_model import dose
import pytest

def test_infusion_range():
    
    infusion_dose = dose.infusion(start=0,
                                  end=30,
                                  rate=15)
    
    assert infusion_dose(t=0) == 15
    assert infusion_dose(t=15.074) == 15
    assert infusion_dose(t=30.004) == 0

@pytest.mark.parametrize(argnames="time, expected_conc",
                         argvalues=[(0, 0),(15.074,0),(42,42)])
def test_bolus_spike(time,expected_conc):
    
    instant_bolus = dose.bolus(amount=42,
                               time=42)
    assert instant_bolus(t=time) == pytest.approx(expected_conc)


def test_negative_timestep():
    
    infusion_dose = dose.infusion(start=0,
                                  end=30,
                                  rate=15)
    
    with pytest.raises(ValueError):
            infusion_dose(t=-2)
            
