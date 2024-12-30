import numpy as np

def gradient_descent(X,y,theta, alpha,iterations):
    m = len(y) # no of training examples
    cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m)*X.T.dot(errors)
        theta -= alpha*gradient # update theta

        # Compute and store the cost
        cost = (1/(2*m))*np.sum(errors**2)
        cost_history.append(cost)
    
    return theta, cost_history

# Exmaple Useage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Add bias term
y = np.array([1, 2, 3])
theta = np.zeros(2)
alpha = 0.1
iterations = 1000

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print("Optimized Parameters:", theta)

# https://chatgpt.com/share/67724efe-60fc-8009-a979-f65c1e3c2000  learn from here for deep info