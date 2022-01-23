import random
def isPrime(number):
    for i in range(3,number):
        if (number%i==0):
            return 0
    return 1
        
a = [random.randint(100,900) for i in range(100)]
even = 0
odd = 0
prime = 0
for i in a:
    if (i%2==0):
        even = even + 1
    else:
        odd = odd + 1
        prime = prime + isPrime(i)
print("List has " + str(even) + " even number(s)" )
print("List has " + str(odd) + " odd number(s)" )
print("List has " + str(prime) + " prime number(s)" )
    
