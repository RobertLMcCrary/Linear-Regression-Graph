import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(0)
square_footage = np.random.rand(100, 1) * 3000  # Square footage of the house
house_prices = 150 + 200 * (square_footage / 1000) + np.random.randn(100, 1) * 50  # House prices with some noise

# Convert the NumPy arrays to PyTorch tensors
square_footage_tensor = torch.tensor(square_footage, dtype=torch.float32)
house_prices_tensor = torch.tensor(house_prices, dtype=torch.float32)


# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(square_footage_tensor)
    loss = criterion(predictions, house_prices_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the results
model.eval()
predicted = model(square_footage_tensor).detach().numpy()
plt.scatter(square_footage, house_prices, label='Original data')
plt.plot(square_footage, predicted, color='red', label='Fitted line')
plt.xlabel('Square Footage')
plt.ylabel('House Prices (in thousands)')
plt.title('Square Footage vs. House Prices')
plt.legend()
plt.show()

# Print model parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.data}')

