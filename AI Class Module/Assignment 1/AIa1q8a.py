class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def describe_restaurant(self):
        print("Restauarant Name - ", self.restaurant_name)
        print("Cuisine Type - ", self.cuisine_type)

    def open_restaurant(self):
        print("Restauarant is open")
        
restaurant = Restaurant("Burger King", "Fast Food")
print(restaurant.restaurant_name)
print(restaurant.cuisine_type)
restaurant.describe_restaurant()
restaurant.open_restaurant()

