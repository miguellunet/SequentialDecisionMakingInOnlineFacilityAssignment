import numpy as np
from miscelaneous import distance_calculator

#Generates a dla instance with num_customers customers evenly spaced in a grid of size grid_size x grid_size. The customers are generated with capacity 1.
def generate_dla_instance(num_customers, grid_size):

    instance = []
    idx = 1

    # Calculate the number of rows and columns based on the desired number of customers
    rows = int(np.ceil(np.sqrt(num_customers)))  # Number of rows based on the square root of num_customers
    cols = int(np.ceil(num_customers / rows))   # Number of columns

    # Create a grid of points, spaced evenly, based on num_customers
    x_step = grid_size / (cols - 1) if cols > 1 else 1  # Step size in x direction (200 is the width of -100 to +100)
    y_step = grid_size / (rows - 1) if rows > 1 else 1  # Step size in y direction (200 is the height of -100 to +100)

    for i in range(rows):
        for j in range(cols):
            # Calculate coordinates based on step size and range of -100 to 100
            x = -grid_size/2 + i * x_step
            y = -grid_size/2 + j * y_step
            if len(instance) < num_customers:
                instance.append([idx, x, y, 1])  # Add customer with capacity 1
                idx += 1

    return instance

# Assigns an expected demand (fractional) to each region, corresponding to a customer in the center of that region. The regions are generated as a grid of size num_regions x num_regions, and the expected demand is calculated as the number of customers left divided by the number of regions.
# Given an instance and a new customer location, removes the closest customer of the instance and replaces it with the new customer location. Returns the updated instance and a copy of the instance before the update.
def generate_regions_instance(customers_left, num_regions, grid_size, new_customer_location):

    instance = []

    # Calculate the number of rows and columns based on the desired number of customers
    rows = int(np.sqrt(num_regions))  # Number of rows based on the square root of num_customers
    cols = int(num_regions / rows)   # Number of columns

    # Create a grid of points, spaced evenly, based on num_customers
    x_step = grid_size / cols   # Step size in x direction (200 is the width of -100 to +100)
    y_step = grid_size / rows   # Step size in y direction (200 is the height of -100 to +100)

    if new_customer_location != None:
        demand_by_location = (customers_left-1)/(rows*cols)
        instance.append([0, new_customer_location[0], new_customer_location[1], 1])  # Add customer with capacity 1
        idx = 1
    else:
        demand_by_location = customers_left/(rows*cols)
        idx = 0

    for i in range(rows):
        for j in range(cols):
            # Calculate coordinates based on step size and range of -100 to 100
            x = -grid_size/2 + x_step/2 + i * x_step
            y = -grid_size/2 + y_step/2 + j * y_step
            instance.append([idx, x, y, demand_by_location])  # Add customer with capacity 1
            idx += 1

    return instance

def update_instance_location(instance, new_customer_location):
    
    #Remove the closest customer of the instance and replace it with the new customer location
    closest_customer = -1
    min_distance = np.inf
    for i in range(len(instance)):
        distance = distance_calculator(new_customer_location, [instance[i][1], instance[i][2]])
        if distance < min_distance:
            min_distance = distance
            closest_customer = i

    instance.pop(closest_customer)

    instance_post_action = instance.copy()

    # Set the first customer to be the new customer (it is useful to retrieve the new customer location in the perfect hindsight function)
    instance.insert(0, [0, new_customer_location[0], new_customer_location[1], 1])

    return instance, instance_post_action