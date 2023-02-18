import numpy as np
import matplotlib.pyplot as plt

from skimage.viewer import CollectionViewer
import time


def choose_sample(segment1, segment2):
    # This turns on interactive mode in matplotlib
    plt.ion()

    # This creates a new sequence of axes
    axes = AxesSequence()

    # This loops over each row in the first segment
    for i in range(segment1.shape[0]):
        
        # This creates a new pair of axes
        ax1, ax2 = axes.new()
        
        # This shows the i-th frame of the first segment on the first axis
        ax1.imshow(segment1[i, :, :, :])
        
        # This sets the title of the first axis
        ax1.set_title('First Segment - {}th frame'.format(i))
        
        # This shows the i-th frame of the second segment on the second axis
        ax2.imshow(segment2[i, :, :, :])
        
        # This sets the title of the second axis
        ax2.set_title('Second Segment - {}th frame'.format(i))

    # This shows the sequence of axes
    axes.show()

    # This prompts the user to choose either segment 1 or segment 2
    user_preference = input("Choose either segment 1 or segment 2:")

    # This closes all the figures
    plt.close('all')

    # This checks the user's choice and assigns res accordingly
    if user_preference == 1:
        res = 1, 0
    elif user_preference == 2:
        res = (0, 1)
    else:
        res = 0.5, 0.5

    # This returns the res value
    return res


# Define a new class named AxesSequence
class AxesSequence(object):
    # Initialize the class with the following properties
    def __init__(self):
        # Create a new figure object with a size of 16x16 inches
        self.fig = plt.figure(figsize=(16, 16))
        # Create an empty list to store the axes
        self.axes = []
        # Set the current axes index to 0
        self._i = 0
        # Set the last created axes index to 0
        self._n = 0
        # Connect the 'key_press_event' event to the on_keypress() method
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    # Define an iterator for the class
    def __iter__(self):
        # Continue indefinitely
        while True:
            # Yield a new set of axes
            yield self.new()

    # Define a method for creating new axes
    def new(self):
        # Create a new subplot with a 2x1 grid and the first subplot as the active one
        ax1 = self.fig.add_subplot(211, visible=False, label=self._n)
        # Create a new subplot with a 2x1 grid and the second subplot as the active one
        ax2 = self.fig.add_subplot(212, visible=False, label=self._n)
        # Increment the last created axes index
        self._n += 1
        # Add the new axes to the axes list
        self.axes.append((ax1, ax2))
        # Return the new axes
        return ax1, ax2

    # Define a method to handle arrow key presses
    def on_keypress(self, event):
        # If the right arrow key is pressed
        if event.key == 'right':
            # Move to the next plot
            self.next_plot()
        # If the left arrow key is pressed
        elif event.key == 'left':
            # Move to the previous plot
            self.prev_plot()
        # Otherwise, return
        else:
            return
        # Redraw the figure canvas
        self.fig.canvas.draw()

    # Define a method to move to the next plot
    def next_plot(self):
        # If the current index is less than the number of axes
        if self._i < len(self.axes):
            # Set the visibility of the current first subplot to false
            self.axes[self._i][0].set_visible(False)
            # Set the visibility of the next first subplot to true
            self.axes[self._i+1][0].set_visible(True)

            # Set the visibility of the current second subplot to false
            self.axes[self._i][1].set_visible(False)
            # Set the visibility of the next second subplot to true
            self.axes[self._i+1][1].set_visible(True)
            # Increment the current index
            self._i += 1

    # Define a method to move to the previous plot
    def prev_plot(self):
        # If the current index is greater than 0
        if self._i > 0:
            # Set the visibility of the current first subplot to false
            self.axes[self._i][0].set_visible(False)
            # Set the visibility of the previous first subplot to true
            self.axes[self._i-1][0].set_visible(True)

            # Set the visibility of the current second subplot to false
            self.axes[self._i][1].set_visible(False)
            # Set the visibility of the previous second subplot to true
            self.axes[self._i-1][1].set_visible(True)
            # Decrement the current index
            self._i -= 1

    # Define a method to show the first plot
    def show(self):
        # Set the visibility of the first subplot to true
        self.axes[0][0].set_visible(True)
        # Set the visibility of the second subplot to true
        self.axes[0][1].set_visible(True)
        # Show the plot
        plt.show()
