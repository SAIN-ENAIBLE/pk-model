"""
Equation builders for each administration regime:
- IV: Intravenous.
- SC: Subcutaneous.
"""

def IV() -> None:
    
    dqc_dt = Dose
    return dqc_dt

def SC() -> None:
    
    dq0_dt = Dose - ka*q0
    dqc_dt = kaq0
    
    return (dq0_dt, dqc_dt)