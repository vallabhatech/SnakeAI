"""
Training Progress Visualization

This module provides real-time visualization of the training progress for the
Snake AI agent. It displays both individual game scores and running mean scores
to help track learning progress over time.
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    """Display real-time training progress with score tracking.
    
    Creates and updates a live matplotlib plot showing the training progress
    by plotting individual game scores and the running mean score. This provides
    visual feedback on how the agent is improving over time.
    
    Args:
        scores (list): List of individual game scores for each episode
        mean_scores (list): List of running mean scores up to each episode
    
    Note:
        - Uses interactive mode (plt.ion()) for real-time updates
        - Clears previous output before each update for smooth animation
        - Shows current values as text labels on the plot
        - Briefly pauses (0.1s) to allow display update
    
    The plot shows:
        - Blue line: Individual game scores (high variance)
        - Orange line: Running mean scores (smoother trend indicator)
        - Text labels: Current score and mean score values
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)