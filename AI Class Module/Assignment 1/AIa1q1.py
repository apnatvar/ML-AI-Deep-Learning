def highest(subject):
    max = 1
    for marks in subject:
        if (marks>max):
            max = marks
    return (max)
def lowest(subject):
    min = 100
    for marks in subject:
        if (marks<min):
            min = marks
    return (min)
def average(subject):
    avg = 0
    for marks in subject:
        avg = avg + marks
    return (avg/5.0)
        

math = [10,20,30,40,50]
science = [20,30,40,50,10]
english = [30,40,50,10,20]
IT = [40,50,10,20,30]
print highest(math)
print highest(science)
print highest(english)
print highest(IT)
print lowest(math)
print lowest(science)
print lowest(english)
print lowest(IT)
print average(math)
print average(science)
print average(english)
print average(IT)
avg = average(math)+average(science)+average(english)+average(IT)
avg = avg/4.0
print avg
