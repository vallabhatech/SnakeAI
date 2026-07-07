"""
Snake Game Environment for Reinforcement Learning

This module implements a Pygame-based Snake game environment designed for
training reinforcement learning agents. It provides the game mechanics,
state transitions, collision detection, and reward system needed for
an AI agent to learn through interaction.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    """Enumeration of possible snake movement directions.
    
    The four cardinal directions that the snake can move in the game.
    These are used to track the snake's current orientation and to
    calculate valid turns.
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 100

class SnakeGameAI:
    """Pygame-based Snake environment for AI training.
    
    This class implements the complete Snake game mechanics including
    movement, collision detection, food placement, and reward calculation.
    It's designed to interface with reinforcement learning agents by
    providing discrete action steps and state information.
    
    Attributes:
        w (int): Width of the game window in pixels (default: 640)
        h (int): Height of the game window in pixels (default: 480)
        display (pygame.Surface): The game display surface
        clock (pygame.time.Clock): Clock for controlling game speed
        direction (Direction): Current movement direction of the snake
        head (Point): Current position of the snake's head
        snake (list[Point]): List of Points representing snake body segments
        score (int): Current score (number of food items eaten)
        food (Point): Current position of the food
        frame_iteration (int): Counter for frames elapsed in current episode
    """

    def __init__(self, w=640, h=480):
        """Initialize the Snake game environment.
        
        Args:
            w (int): Width of the game window in pixels (default: 640)
            h (int): Height of the game window in pixels (default: 480)
        """
        #init display
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        """Reset the game to initial state for a new episode.
        
        Resets the snake to starting position (center of screen, facing right),
        initializes score to zero, places new food, and resets the frame counter.
        This is called at the start of each training episode.
        """
        # init default game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # to prevent infinite loops


    def _place_food(self):
        """Place food at a random location on the game grid.
        
        Calculates random coordinates aligned to the grid system and ensures
        food doesn't spawn on the snake's body. Recursively calls itself
        if the generated position conflicts with the snake.
        
        The grid alignment ensures food always appears at valid cell positions
        that the snake can actually reach.
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        """Execute one game step based on the provided action.
        
        This is the main game loop method that processes a single action from
        the AI agent. It handles movement, collision detection, food collection,
        and UI updates. Returns the reward, game over status, and current score.
        
        Args:
            action (list): Action array representing movement choice
                          [1,0,0] = straight, [0,1,0] = right, [0,0,1] = left
        
        Returns:
            tuple: (reward, game_over, score)
                - reward (int): +10 for food, -10 for collision, 0 otherwise
                - game_over (bool): True if episode should end
                - score (int): Current number of food items collected
        
        Note:
            Also terminates if frame_iteration > 100 * len(snake) to prevent
            infinite loops where the snake avoids food indefinitely.
        """
        self.frame_iteration += 1
        # 1. handle user quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move(remove tail)
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        """Check if a point collides with walls or the snake's body.
        
        Detects collision with screen boundaries or the snake's own body.
        Used both for game over detection and for state representation
        (checking if nearby positions are dangerous).
        
        Args:
            pt (Point, optional): Point to check for collision. 
                                 Defaults to snake head if not provided.
        
        Returns:
            bool: True if collision detected, False otherwise
        
        Note:
            When checking for self-collision, the head (index 0) is excluded
            since we check against the rest of the body.
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        """Render the current game state to the display.
        
        Draws the background, snake (with visual styling), food, and current score.
        The snake is rendered with a two-tone design (outer and inner squares)
        for better visual appeal and depth perception.
        
        The UI update happens at the end of each game step after all logic
        has been processed.
        """
        # set background
        self.display.fill(BLACK)
        
        
        # draw snake
        for pt in self.snake:
            # outer square
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # inner square
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()


    def _move(self, action):
        """Update snake position based on the provided action.
        
        Converts relative actions (straight, right, left) into absolute
        directions by considering the current orientation. Uses a clockwise
        direction array to calculate turns correctly regardless of current
        direction.
        
        Args:
            action (list): Action array [1,0,0] for straight, [0,1,0] for right turn,
                          [0,0,1] for left turn
        
        Note:
            This relative action system prevents the snake from making 180-degree
            turns which would cause immediate self-collision. The agent can only
            turn 90 degrees left or right from its current direction.
        """
        # action space = [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # change direction based on action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        # update the head
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)