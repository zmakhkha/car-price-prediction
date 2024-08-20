import numpy as np
import json

def load_parameters(file_path):
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    theta = np.array(parameters['theta']).reshape(-1, 1)
    mean_x = parameters['mean_x']
    std_x = parameters['std_x']
    return theta, mean_x, std_x

def predict_price(mileage, theta, mean_x, std_x):
    normalized_mileage = (mileage - mean_x) / std_x
    x = np.array([[normalized_mileage, 1]])    
    price = x.dot(theta)
    return price[0][0]

def main():
    theta, mean_x, std_x = load_parameters('.model_parameters.json')    
    mileage = float(input("Enter the car mileage: "))
    predicted_price = predict_price(mileage, theta, mean_x, std_x)    
    print(f"Predicted price for mileage {mileage}: ${predicted_price:.2f}")


if __name__ == "__main__":
    main()
