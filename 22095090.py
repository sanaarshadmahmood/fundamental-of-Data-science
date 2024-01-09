
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 1 15:59:25 2024

@author: HJ
"""


# Read data from the data file
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
data = np.loadtxt('data0-1.csv')

# Create a histogram
plt.hist(data, bins=30, density=True, alpha=0.7,
         color='blue', edgecolor='black')

# Fit a normal distribution to the data
mu, std = norm.fit(data)

# Create a probability density function (PDF)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)

# Plot the PDF
plt.plot(x, p, 'k', linewidth=2)

# Calculate the mean annual salary (W̃)
mean_salary = np.mean(data)

# Calculate the mean annual salary using the obtained PDF
mean_salary_pdf = np.trapz(x * p, x)

# Print both values on the graph
plt.text(0.05, 0.95, f'W̃ (from data) = {mean_salary:.2f}', transform=plt.gca(
).transAxes, fontsize=10, verticalalignment='bottom')
plt.text(0.05, 0.95, f'W̃ (from PDF) = {mean_salary_pdf:.2f}', transform=plt.gca(
).transAxes, fontsize=10, verticalalignment='top')

# Calculate the 75th percentile value (X)
# This is the value below which 75% of the salaries lie
X = np.percentile(data, 75)

# Print the 75th percentile value on the graph
plt.text(0.5, 0.9, f'X = {X:.2f}', transform=plt.gca(
).transAxes, fontsize=10, verticalalignment='top')

# Add labels and title
plt.xlabel('Annual Salary')
plt.ylabel('Probability Density')
plt.title('Probability Density Function and Histogram of Annual Salaries')
plt.legend(['PDF', 'Histogram'])

# Show the plot
plt.show()
