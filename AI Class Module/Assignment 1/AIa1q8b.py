class User:
    def __init__(self, first_name, last_name, age, city, email):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.city = city
        self.email = email

    def describe_user(self):
        print("User Profile")
        print("Name - ", self.first_name, self.last_name)
        print("Age - ", self.age)
        print("City - ", self.city)
        print("Email - ", self.email)

    def greet_user(self):
        print("Welcome," + str(self.first_name) + " " + str(self.last_name) + "!! Have a nice day")
        print()

user = User("Apnatva", "Rawat", 20, "Dehradun", "arawat@gmail.com")
user.describe_user()
user.greet_user()
user = User("Apnatva", "Singh", 21, "Mussoorie", "asingh@gmail.com")
user.describe_user()
user.greet_user()

