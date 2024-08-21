import numpy as np
import matplotlib.pyplot as plt
import json

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, xlabel, ylabel, image_path):
    plt.scatter(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Data Visualization')
    plt.savefig(image_path)
    plt.close()
    
def get_data(data_path):
    try:
        data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
    
    x = data[:, 0]
    y = data[:, 1]
    
    # Reshape x and y to column vectors (if needed)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # Plot before normalization
    plot(x[:, 0], y[:, 0], 'Mileage', 'Price', 'before_normalization.png')
    
    # Normalize x
    mean_x = np.mean(x)
    std_x = np.std(x)
    x = (x - mean_x) / std_x
    
    # Add a column of ones for the intercept term
    ones = np.ones(x.shape)
    x = np.hstack((x, ones))
    
    # Plot after normalization
    plot(x[:, 0], y[:, 0], 'Normalized Mileage', 'Price', 'after_normalization.png')
    
    return x, y, mean_x, std_x


def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    prediction_error = model(X, theta) - y
    gradient = (1/m) * X.T.dot(prediction_error)
    return gradient

def gradient_descente(X, y, theta, learning_rate, iterations):
    for _ in range(iterations):
        gradient = grad(X, y, theta)
        theta = theta - learning_rate * gradient
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            raise ValueError("Theta contains NaN or infinity values.")
    return theta

def plot_with_model(x_original, y, theta, mean_x, std_x, xlabel, ylabel, image_path, title):
    # Plot the original data
    plt.scatter(x_original, y, color='blue', marker='o', label='Data')

    # Plot the model line in the original scale
    x_plot = np.linspace(np.min(x_original), np.max(x_original), 100)
    x_plot_normalized = (x_plot - mean_x) / std_x  # Normalize x for model prediction
    y_plot = theta[0] + theta[1] * x_plot_normalized
    plt.plot(x_plot, y_plot, color='red', label='Model')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(image_path)
    plt.close()

def main():
    x, y, mean_x, std_x = get_data('data.csv')
    theta = np.zeros((2,1))
    plot_with_model(x[:,0], y, theta, 'Mileage', 'Price', 'Before_training.png', 'Before_training')

    final_theta = gradient_descente(x, y, theta, learning_rate=0.01, iterations=1000)
    
     # Plot after training
    plot_with_model(x_original, y, final_theta, mean_x, std_x, 'Mileage', 'Price', 'after_training.png', 'Model After Training')

    print(f"Final theta: {final_theta}")
    parameters = {
        'theta': final_theta.flatten().tolist(),
        'mean_x': float(mean_x),
        'std_x': float(std_x)
    }

    with open('.model_parameters.json', 'w') as f:
        json.dump(parameters, f)
    

if __name__ == "__main__":
    main()