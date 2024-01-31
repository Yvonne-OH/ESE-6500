# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:38:39 2024

@author: 39829
"""

import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        alpha = np.zeros((len(self.Observations), len(self.Transition)))

        # Base case
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]]

        # Recursive case
        for t in range(1, len(self.Observations)):
            for j in range(len(self.Transition)):
                alpha[t, j] = self.Emission[j, self.Observations[t]] * np.sum(alpha[t-1, :] * self.Transition[:, j])

        return alpha

    def backward(self):
        beta = np.zeros((len(self.Observations), len(self.Transition)))

        # Base case
        beta[-1, :] = 1

        # Recursive case
        for t in range(len(self.Observations) - 2, -1, -1):
            for i in range(len(self.Transition)):
                beta[t, i] = np.sum(beta[t+1, :] * self.Transition[i, :] * self.Emission[:, self.Observations[t+1]])

        return beta

    def gamma_comp(self, alpha, beta):
        gamma = np.multiply(alpha, beta)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.zeros((len(self.Observations) - 1, len(self.Transition), len(self.Transition)))

        for t in range(len(self.Observations) - 1):
            denominator = np.dot(alpha[t, :].T, beta[t + 1, :] * self.Emission[:, self.Observations[t + 1]].T)
            for i in range(len(self.Transition)):
                numerator = alpha[t, i] * self.Transition[i, :] * self.Emission[:, self.Observations[t + 1]] * beta[t + 1, :]
                xi[t, i, :] = numerator / denominator

        return xi

    def update(self, gamma, xi):
        # Update initial state distribution
        new_init_state = gamma[0, :]
        # Update transition matrix
        T_prime = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0, keepdims=True)

        # Assume the emission matrix M is of shape (N_states, N_observations)
        # Update emission probabilities (M_prime) if necessary
        M_prime = np.zeros_like(self.Emission)
        for s in range(len(self.Transition)):
            for o in range(self.Emission.shape[1]):
                mask = (self.Observations == o)
                M_prime[s, o] = np.sum(gamma[mask, s]) / np.sum(gamma[:, s])

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, new_init_state, T_prime, M_prime):
        # Compute new alpha with updated HMM parameters
        self.Initial_distribution = new_init_state
        self.Transition = T_prime
        self.Emission = M_prime
        new_alpha = self.forward()

        # Original trajectory probability
        P_original = np.sum(alpha[-1, :])
        # Updated trajectory probability
        P_prime = np.sum(new_alpha[-1, :])

        return P_original, P_prime
    
"""   
    def infer_missing_observations(self):

        inferred_observations = self.Observations.copy()
        missing_indices = np.where(self.Observations == 2)[0]
        
        # Compute gamma using forward and backward algorithms
        alpha = self.forward()
        beta = self.backward()
        gamma = self.gamma_comp(alpha, beta)
        
        # Infer the missing observations by taking the state with the highest gamma probability
        for idx in missing_indices:
            most_probable_state = np.argmax(gamma[idx])
            inferred_observations[idx] = most_probable_state
        
        return inferred_observations

# Example usage
# Define your transition matrix, emission matrix, initial distribution, and observations.
# Replace these with your actual matrices and observation sequence.
Transition = np.array([[0.5, 0.5], [0.5, 0.5]])
Emission = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
Initial_distribution = np.array([0.5, 0.5])
Observations = np.array([2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1])   # Placeholder for actual observations

# Create an instance of the HMM
hmm = HMM(Observations, Transition, Emission, Initial_distribution)

# Run the forward and backward algorithms
alpha = hmm.forward()
beta = hmm.backward()

# Compute gamma and xi
gamma = hmm.gamma_comp(alpha, beta)
xi = hmm.xi_comp(alpha, beta, gamma)

# Update the HMM parameters
T_prime, M_prime, new_init_state = hmm.update(gamma, xi)

# Compute trajectory probabilities
P_original, P_prime = hmm.trajectory_probability(alpha, new_init_state, T_prime, M_prime)

inferred_observations = hmm.infer_missing_observations()
# Mapping from numbers to strings
location_mapping = {0: 'LA', 1: 'NY'}

# Convert the array of 0s and 1s to an array of strings
inferred_observations = [location_mapping[location] for location in inferred_observations]
print("Inferred observations:", inferred_observations)

#%%


# Create an instance of the HMM with the initial parameters
hmm = HMM(Observations, Transition, Emission, Initial_distribution)

# Run the forward algorithm with the initial parameters
alpha = hmm.forward()

# Compute the initial trajectory probability
P_lambda = np.sum(alpha[-1, :])

# Update the HMM parameters using Baum-Welch (gamma and xi have already been calculated)
T_prime, M_prime, new_init_state = hmm.update(gamma, xi)

# Set the updated parameters in the HMM
hmm.Transition = T_prime
hmm.Emission = M_prime
hmm.Initial_distribution = new_init_state

# Run the forward algorithm with the updated parameters
new_alpha = hmm.forward()

# Compute the updated trajectory probability
P_lambda_prime = np.sum(new_alpha[-1, :])

# Print the two probabilities to compare
print("P(Y | λ):", P_lambda)
print("P(Y | λ'):", P_lambda_prime)

# Check if the updated probability is greater than the initial
if P_lambda_prime > P_lambda:
    print("The updated model better fits the observation sequence.")
else:
    print("The updated model does not fit the observation sequence better.")"""
