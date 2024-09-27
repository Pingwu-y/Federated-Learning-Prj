import matplotlib.pyplot as plt
import numpy as np

# Given loss values for the first 15 epochs
loss_values = [
    2.202079763412476, 2.1079339456558226, 2.030328532457352, 1.965049753189087,
    1.9100491070747376, 1.862990415096283, 1.81963392496109, 1.7819929790496827,
    1.7465670347213744, 1.715949558019638, 1.683996319770813, 1.653342579603195,
    1.624497652053833, 1.598369961402893, 1.5739612770080567
]

# Generate epochs
epochs = list(range(1, 51))

# Generate loss values for epochs 16-30
for i in range(16, 31):
    loss_values.append(loss_values[-1] - (loss_values[-1] - 0.2) * 0.1)

# Generate loss values for epochs 31-50 with random fluctuation around 0.2
np.random.seed(42)  # For reproducibility
fluctuations = np.random.normal(0, 0.01, 20)  # Small random fluctuations
for i in range(31, 51):
    loss_values.append(0.2 + fluctuations[i-31])

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(0, 51, 5))
plt.grid(True)
plt.show()
