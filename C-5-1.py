learning_rate = 0.1
epochs = 10000

for i in range(epochs):
    # Forward pass
    A1, A2 = forward(X)
    
    # Compute loss (Mean Squared Error)
    loss = np.mean((y - A2)**2)
    
    # Backward pass
    dA2 = (y - A2) * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)
    
    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if i % 1000 == 0:
        print(f"Epoch [{i}], Loss: {loss:f}")
_, A2 = forward(X)
predictions = (A2 > 0.5).astype(int)
print("Predictions: \n", predictions)