def computeCI(principal, roi, time):
    ci = (principal*pow((1+(roi/100)), time))-principal
    return round(ci, 2)

