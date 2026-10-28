def infusion(start, end, rate):
    
    def dose(t):
        if start <= t < end:
            return rate
        else:
            return 0
    
    return dose

def bolus(amount, time):
    
    pass
