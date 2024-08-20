import numpy as np
import json

def get_data(data_path):
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    x = data[:,0]
    y = data[:,1]
    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    mean_x = np.mean(x)
    std_x = np.std(x)
    x = (x - mean_x) / std_x
    ones = np.ones(x.shape)
    x = np.hstack((x, ones))
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

def main():
    x, y, mean_x, std_x = get_data('data.csv')
    theta = np.zeros((2,1))

    final_theta = gradient_descente(x, y, theta, learning_rate=0.01, iterations=1000)
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