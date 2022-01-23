from data import computeCI
principal = float(input("Enter Principal Amount - "))
roi = float(input("Enter Rate of Interest (per annum) - "))
time = float(input("Enter Time (in years) - "))
print("The Compound Interest for the amount is " + str(computeCI(principal, roi, time)))

