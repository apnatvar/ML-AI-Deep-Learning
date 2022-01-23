basic_salary = 1000
if (basic_salary<=10000):
    hra = 0.2
    da = 0.8
elif(basic_salary<=20000):
    hra = 0.25
    da = 0.9
else:
    hra = 0.3
    da = .95
gross_salary = basic_salary + basic_salary * hra + basic_salary * da
print gross_salary
