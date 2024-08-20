import numpy as np

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

x = data[:,0]
y = data[:,1]

print("Before reshaping")
print(x.shape)
print(y.shape)

print("After reshaping")
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)
print(x.shape)
print(y.shape)


ones = np.ones(x.shape)
print("ones vector shape : ")
print(ones.shape)
x = np.hstack((x, ones))
print("new x shape after stacking : ")
print(x.shape)
print(x)

theta = np.zeros((2,1))
print(f"theta : {theta}")


def model(X, theta):
    return x.dot(theta)

def cost_function(x, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum ((model(x,theta) - y)**2)

print(cost_function(x, y, theta))


def grad(x, y, theta):
    m = len(y)
    prediction_error = model(x, theta) - y
    print(f"Prediction error: {prediction_error}")  # Debugging output
    
    gradient = (1/m) * x.T.dot(prediction_error)
    print(f"Gradient: {gradient}")  # Debugging output
    
    return gradient


print("dssdfsdfsdfsdf",theta)

def gradient_descente(x, y, theta, learning_rate, iterations):
    print('----------------')
    print(theta - learning_rate * grad(x, y, theta))
    for _ in range(iterations):
        theta = theta - learning_rate * grad(x, y, theta)
    return theta

final_theta = gradient_descente(x, y, theta, learning_rate=0.0001, iterations=1000)
print(final_theta)