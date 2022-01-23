password = raw_input("Enter Password: ")
lower = False
upper = False
special_character = False
digit = False
if (len(password)>=6 and len(password)<=16):
    for i in password:
        if (i.islower()):
            lower = True
        if (i.isupper()):
            upper = True
        if (i.isdigit()):
            digit = True
        if (i=='@' or i=='$' or i=='_'):
            special_character = True
if (lower and upper and special_character and digit):
    print("Valid Password")
else:
    print("Invalid Password")
