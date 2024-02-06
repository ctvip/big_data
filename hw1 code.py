import numpy as np
import matplotlib as plt
# Given linear equation parameters
intercept = 18/35
slope = 62/35

# Generate 100 random x values between the smallest and largest x in the original data
x_random = np.linspace(min(x_values), max(x_values), 100)

# Generate y values from the linear equation and add Gaussian noise
noise_std = 0.5  # Standard deviation of the Gaussian noise
y_random = intercept + slope * x_random + np.random.normal(0, noise_std, x_random.size)

# Compute the least squares estimate using numpy's polyfit function for a degree 1 polynomial
slope_new, intercept_new = np.polyfit(x_random, y_random, 1)

# Generate y values using the new least squares estimate
y_new_line = intercept_new + slope_new * x_random

# Plot the original data, the original line, and the new dataset with the new line
plt.figure(figsize=(10, 6))
plt.scatter(x_random, y_random, color='green', label='Random Data with Noise', alpha=0.5)
plt.plot(x_random, y_new_line, color='orange', label='New Least Squares Line')
plt.plot(x_values, equation_y_values, color='red', label='Original Line', linestyle='--')

# Set the plot title and labels
plt.title('Original vs New Data and Fitted Lines')
plt.xlabel('x')
plt.ylabel('y')

# Show legend
plt.legend()

# Show the plot with a grid
plt.grid(True)
plt.show()

# Return the parameters of the new line for verification
slope_new, intercept_new