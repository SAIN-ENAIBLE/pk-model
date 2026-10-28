def infusion(start, end, rate):
    
    def dose(t):
        if t < 0:
            raise ValueError(f"Time cannot be negative (t={t})")
        if start <= t < end:
            return rate
        
        return 0
    
    return dose

def bolus(amount, time):
    
    pass
