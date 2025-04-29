import random
import numpy as np
import copy
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import os




# ANSI color codes
BLUE = '\033[94m'
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

class Maze:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.maze = np.zeros((n, n), dtype=object)
        self.walls = set()
        self.goal = None
        self.start = None
        self.qmatrix = np.zeros((n*n, 4))
        
        
        # Optimized hyperparameters
        self.initial_epsilon = 1.0  # Start with full exploration
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate per episode
        
        self.total_episodes = 20000
        self.alpha = 0.1  # Reduced learning rate for stability
        self.gamma = 0.99  # Slightly increased discount factor
        
        self.action = [(0,1), (0,-1), (-1,0), (1,0)]
        self.action_icon = ['→','←','↑','↓']
        self.cumulative_rewards = [0 for i in range(self.total_episodes)]
   
    def create_maze(self):


        self.walls.clear()
        self.goal = None
        self.start = None


        while len(self.walls) < self.m:
            wall = (random.randint(0, self.n-1), random.randint(0, self.n-1))
            if wall not in self.walls:
                self.walls.add(wall)
        
        while self.goal is None or self.goal in self.walls:
            self.goal = (random.randint(0, self.n-1), random.randint(0, self.n-1))

        while self.start is None or self.start in self.walls or self.start == self.goal:
            self.start = (random.randint(0, self.n-1), random.randint(0, self.n-1))
    
    def reward_func(self, state):
        if state == self.goal:
            return 10
        if state in self.walls:
            return -10
        return -1
   
    def get_state_num(self, state):
        return state[0] * self.n + state[1]
   
    def get_state_loc(self, state_num):
        return (state_num // self.n, state_num % self.n)
   
    def is_valid_move(self, state):
        return (0 <= state[0] < self.n and 
                0 <= state[1] < self.n and 
                state not in self.walls)
   
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            state_num = self.get_state_num(state)
            return np.argmax(self.qmatrix[state_num])
     
    def sarsa_max(self):
        for episode in range(self.total_episodes):
            episode_reward = 0
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, 
                               self.initial_epsilon * (self.epsilon_decay ** episode))
            
            current_state_loc = self.start
            current_state = self.get_state_num(current_state_loc)
            counter = 0

            while current_state_loc != self.goal and counter<self.n*self.n:
                action = self.choose_action(current_state_loc)
                
                next_state_loc = (
                    current_state_loc[0] + self.action[action][0],
                    current_state_loc[1] + self.action[action][1]
                )
                
                if self.is_valid_move(next_state_loc):
                    
                    reward = self.reward_func(next_state_loc)
                    next_state = self.get_state_num(next_state_loc)
                    
                    self.qmatrix[current_state][action] += self.alpha * (
                        reward + 
                        self.gamma * np.max(self.qmatrix[next_state]) - 
                        self.qmatrix[current_state][action]
                    )
                    
                    current_state = next_state
                    current_state_loc = next_state_loc
                    episode_reward+=reward

                else:
                    
                    self.qmatrix[current_state][action] += self.alpha * (
                        -1 + self.gamma * np.max(self.qmatrix[current_state]) - 
                        self.qmatrix[current_state][action]
                    )
                    episode_reward+=-1
                
                counter+=1
    
            self.cumulative_rewards[episode]=self.cumulative_rewards[episode-1]+episode_reward if episode>0 else episode_reward
            # self.cumulative_rewards[episode] = episode_reward
            
    def fetch_policy(self):
        for i in range(self.n):
            for j in range(self.n):
                state_num = self.get_state_num((i, j))
                if (i, j) == self.start:
                    self.maze[i, j] = f'{BLUE}{self.action_icon[np.argmax(self.qmatrix[state_num])]}{RESET}'
                elif (i, j) not in self.walls and (i, j) != self.goal:
                    self.maze[i, j] = self.action_icon[np.argmax(self.qmatrix[state_num])]
                elif (i, j) == self.goal:
                    self.maze[i, j] = f'{GREEN}G{RESET}'
                else:
                    self.maze[i, j] = f'{RED}W{RESET}'
    
    def print_maze(self):
        for row in self.maze:
            print(' '.join(map(str, row)))

    def is_solvable(self):
        visited = set()
        stack = [self.start]
        
        while stack:
            current = stack.pop()
            
            if current == self.goal:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for dx, dy in self.action:
                next_state = (current[0] + dx, current[1] + dy)
                if self.is_valid_move(next_state) and next_state not in visited:
                    stack.append(next_state)
        
        return False

    def trace_solution(self):
        
        # Deep copy of the maze matrix
        solved_maze = copy.deepcopy(self.maze)

        current_state_loc = self.start
        steps = []
        visited = set()

        while current_state_loc != self.goal:
            if current_state_loc in visited:
                print("Maze did not solve correctly.")
                return
            
            visited.add(current_state_loc)
            steps.append(current_state_loc)

            # Get the best action from the policy
            current_state = self.get_state_num(current_state_loc)
            best_action = np.argmax(self.qmatrix[current_state])

            # Move to the next state
            next_state_loc = (
                current_state_loc[0] + self.action[best_action][0],
                current_state_loc[1] + self.action[best_action][1]
            )

            if not self.is_valid_move(next_state_loc):
                print("Maze did not solve correctly.")
                return

            current_state_loc = next_state_loc

        # If the goal is reached
        steps.append(self.goal)
        print("Maze solved successfully!")

        # Highlight the solution path
        for step in steps:
            i, j = step
            if step == self.start:
                solved_maze[i, j] = f'\033[1;34mS{RESET}'  # Start in bold blue
            elif step == self.goal:
                solved_maze[i, j] = f'\033[1;32mG{RESET}'  # Goal in bold green
            else:
                solved_maze[i, j] = f'\033[1;33m{'*'}\033[0m'  # Path in bold yellow
        
        # Print the solved maze
        print("\nSolved Maze with Path Highlighted:")
        for row in solved_maze:
            for item in row:
                if item not in self.action_icon:
                    print(item,end=' ')
                else :
                    print('.',end=' ')
            print()
        return steps
    
    def plot_cumulative_rewards(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cumulative_rewards, label="Cumulative Rewards", color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Rewards")
        plt.title("Cumulative Rewards Over Episodes")
        plt.legend()
        plt.grid()
        plt.show()
    
    def animate_solution(self,steps):
        """Animate the solution path based on the learned policy."""
        if not steps:
            print("No solution found to animate.")
            return

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.grid(True)
        ax.invert_yaxis()
        # ax.invert_xaxis()


        # Plot walls and goal
        for wall in self.walls:
            ax.add_patch(plt.Rectangle(wall[::-1], 1, 1, color="black"))
        ax.add_patch(plt.Rectangle(self.goal[::-1], 1, 1, color="green"))
        ax.add_patch(plt.Rectangle(self.start[::-1], 1, 1, color="blue"))


        # Agent marker
        agent, = ax.plot([], [], 'ro', markersize=10)

        def update(frame):
            if frame >= len(steps):
                print(f"Frame {frame} out of bounds for steps.")
                return []
            x, y = steps[frame]
            agent.set_data([y + 0.5], [x + 0.5])  # Wrap x and y in lists
            return agent,



        ani = FuncAnimation(fig, update, frames=len(steps), repeat=False)
        plt.show()
    
    def plot_q_heatmaps(self):
        """Plot heatmaps for Q-values for each action."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, ax in enumerate(axes):
            q_values = self.qmatrix[:, i].reshape(self.n, self.n)
            im = ax.imshow(q_values, cmap="coolwarm", origin="upper")
            ax.set_title(f"Q-values for action {self.action_icon[i]}")
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

def main():
    clear_terminal()
    maze_dim = 20
    wall_count = 50

    maze = Maze(maze_dim,wall_count)
 

    maze.create_maze()
    while(not maze.is_solvable()):
        maze.create_maze()

  

   
    maze.sarsa_max()
   
    maze.fetch_policy()
    
    print("Extracted Policy : \n")
    maze.print_maze()
   
    print('\n')
    steps =  maze.trace_solution()

    maze.plot_cumulative_rewards()
    maze.plot_q_heatmaps()
    maze.animate_solution(steps)

if __name__ == "__main__":
    main()