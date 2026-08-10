import numpy as np
import json

def distance_calculator(location1, location2):
    return np.sqrt((location1[0] - location2[0])**2 + (location1[1] - location2[1])**2) #Euclidean distance

def read_instance(instance_file):
    # loads a pre-generated, fixed order sequence (see instances_generator.py) so every policy can be evaluated on identical realized demand.
    with open(instance_file, 'r') as f:
        data = json.load(f)  # Load JSON file
    return data["instances"]