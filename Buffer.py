import numpy as np
import random

# Define a class called "experience_buffer"
class experience_buffer():
    # Define a constructor method
    def __init__(self, buffer_size=50000):
        # Initialize an empty buffer list
        self.buffer = []
        # Set the buffer size
        self.buffer_size = buffer_size

    # Define a method for adding experiences to the buffer
    def add(self, experience):
        # Check if the length of the buffer plus the length of the new experience is greater than or equal to the buffer size
        if len(self.buffer) + len(experience) >= self.buffer_size:
            # Remove the oldest experiences from the buffer until the buffer is at its maximum size
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        # Add the new experience to the buffer
        self.buffer.extend(experience)

    # Define a method for sampling experiences from the buffer
    def sample(self, size):
        # Use the random.sample method to select a random sample of experiences from the buffer
        random_sample = random.sample(self.buffer, size)
        # Reshape the sample to have the dimensions [size, 5]
        reshaped_sample = np.reshape(np.array(random_sample), [size, 5])
        return reshaped_sample
