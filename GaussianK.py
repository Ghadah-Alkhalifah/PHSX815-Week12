import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set up the experiment parameters
Nflips = 25  # Number of coin flips
alpha = 13.8  # Alpha parameter for the beta distribution
beta = 9.2  # Beta parameter for the beta distribution
seed=4444
np.random.seed(seed)

# Sample the bias parameter (alternative hypothises) from a beta distribution (prior distribution)
bias = np.random.beta(alpha, beta, size=Nflips)
prior_mean = np.mean(bias)

# Simulate flipping the coin and count the number of heads
coin_flips = np.random.binomial(1, bias, size=Nflips)
num_H = np.sum(coin_flips)

# Compute the posterior distribution of the bias parameter
posterior_alpha =  num_H+ alpha
posterior_beta =  Nflips +beta- num_H
posterior_dist = np.random.beta(posterior_alpha, posterior_beta, size=1000)


# Define the Gaussian kernel density estimation
def gaussian_ker(Data, bandwidth):
    gkde = gaussian_kde(Data, bw_method=bandwidth)
    return gkde

# Create subplots to show best bandwidth value
plt.figure(figsize=(6, 6))

# Plot the estimated density 
bandwidth_list = [0.1, 0.3, 0.5, 0.9]
for j, bandwidth in enumerate(bandwidth_list):
    kde_density = gaussian_ker(posterior_dist, bandwidth)
    xVal = np.linspace(.3, 1, 100)
    plt.subplot( 2, 2,j+1)
    plt.hist(posterior_dist, bins=25, density=True, alpha=0.9, color='green')
    plt.plot(xVal, kde_density(xVal), linewidth=1.75, color='red')
    plt.title('Gaussian Kernel Density Estimation (bandwidth={})'.format(bandwidth))

plt.tight_layout()
plt.show()
