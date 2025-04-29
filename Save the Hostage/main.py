import pygame
import random
import math
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
TILE_SIZE = 40
ROWS, COLS = HEIGHT // TILE_SIZE, WIDTH // TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rescue the Hostage - Local Search")

# Colors
WHITE = (240, 248, 255)
RED = (255, 69, 0)      # Hostage color
BLUE = (30, 144, 255)   # Player color
LIGHT_GREY = (211, 211, 211) # Background grid color
FLASH_COLOR = (50, 205, 50) # Victory flash color
BUTTON_COLOR = (50, 205, 50) # Button color
BUTTON_TEXT_COLOR = (255, 255, 255) # Button text color

# Load images for player, hostage, and walls
player_image = pygame.image.load("AI1.png")  
hostage_image = pygame.image.load("AI2.png")  
wall_images = [
    pygame.image.load("AI3.png"),
    pygame.image.load("AI4.png"),
    pygame.image.load("AI5.png")
]

# Resize images to fit the grid
wall_images = [pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE)) for img in wall_images]
player_image = pygame.transform.scale(player_image, (TILE_SIZE, TILE_SIZE))
hostage_image = pygame.transform.scale(hostage_image, (TILE_SIZE, TILE_SIZE))

# Constants for recent positions
MAX_RECENT_POSITIONS = 50
GENERATION_LIMIT = 50
MUTATION_RATE = 0.1

# Function to generate obstacles
def generate_obstacles(num_obstacles):
    obstacles = []
    while len(obstacles) < num_obstacles:
        new_obstacle = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        if new_obstacle not in obstacles:  # Make sure obstacles are not overlapping
            obstacles.append(new_obstacle)
    obstacle_images = [random.choice(wall_images) for _ in obstacles]
    return obstacles, obstacle_images

# Function to start a new game
def start_new_game():
    global player_pos, hostage_pos, recent_positions, obstacles, obstacle_images,recent_positions_gen,kromozom_size

    obstacles, obstacle_images = generate_obstacles(50)
    kromozom_size = (ROWS+COLS)*(len(obstacles)//20)
    recent_positions = []
    recent_positions_gen=  []

    # Generate player and hostage positions with a larger distance
    while True:
        player_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        hostage_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        distance = math.dist(player_pos, hostage_pos)
        if distance > 10 and player_pos not in obstacles and hostage_pos not in obstacles:
            break

# Function to move the player closer to the hostage using Hill Climbing algorithm
def hill_climbing(player_pos, hostage_pos, obstacles):
    neighbors = [[-1,0],[0,-1],[1,0],[0,1]]
    
    player_distance = math.dist(player_pos,hostage_pos)
    min_dist = player_distance
    selected_nghbr = player_pos

    for item in neighbors:
       item[0]+=player_pos[0]
       item[1]+=player_pos[1]
       
       if item[0]>=0 and item[0]<COLS and item[1]>=0 and item[1]<ROWS:
           if item not in obstacles:
               nghbr_distance = math.dist(item,hostage_pos)
               if(nghbr_distance<min_dist):
                  min_dist = nghbr_distance
                  selected_nghbr = item

    return selected_nghbr
##########################################################################################

temperature = 100  # Initial temperature
cooling_rate = 0.97

def get_valid_neighbors(player_pos, obstacles):
    neighbors = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    res = []
    for item in neighbors:
        new_col = item[0] + player_pos[0]
        new_row = item[1] + player_pos[1]
        
        if 0 <= new_col < COLS and 0 <= new_row < ROWS:
            if [new_col, new_row] not in obstacles:
                res.append([new_col, new_row])
    return res



def acceptance_probability(old_cost, new_cost, temp):   
    return math.exp((old_cost - new_cost) / temp)

def simulated_annealing(player_pos, hostage_pos, obstacles):
    global temperature, cooling_rate

    if temperature > 0.0001:
        temperature *= cooling_rate
        
        player_cost = math.dist(player_pos, hostage_pos)
        neighbors = get_valid_neighbors(player_pos, obstacles)
        random_neighbor = random.choice(neighbors) if neighbors else player_pos
        
        neighbor_cost = math.dist(random_neighbor, hostage_pos)
        delta = neighbor_cost - player_cost
        
        if delta < 0:
            # Favorable move
            print(f"Temperature: {temperature}, Move accepted with prob = 1")
            return random_neighbor
        else:
            # Unfavorable move, accept based on probability
            prob = acceptance_probability(player_cost, neighbor_cost, temperature)
            print(f"Temperature: {temperature}, Move accepted with prob = {prob}")
            if random.random() <= prob:
                return random_neighbor
        return player_pos
    else:
        return [-1,-1]    
       
    
###############################################################################################

population_size = 20
generations = 100
kromozom_size = (ROWS+COLS)*(10)



# Fitness function
def fitness(kromozom,player_pos,hostage_pos):
    final_pos = calculate_playes_pos_according_to_kromozom(player_pos,kromozom)
    return math.dist(final_pos,hostage_pos)

def calculate_playes_pos_according_to_kromozom(player_pos,kromozom):
    result = player_pos
    for item in kromozom:
        result= move(item,result)
    return result

def move(move_num,pos):
    if move_num==0:
        return [pos[0]-1,pos[1]]

    elif move_num==1:
        return [pos[0]+1,pos[1]]
    elif move_num==2:
        return [pos[0],pos[1]+1]
    else:
        return [pos[0],pos[1]-1]


# Generate random population
def generate_population(player_pos,obs,hostage_pos):
  global kromozom_size,population_size,generations
  population = []
  for i in range(population_size):
    test_pos = player_pos
    kromozom = []
    for j in range(kromozom_size):
        movmnt = random.randint(0,3)
        next_pos = move(movmnt,test_pos)
        while not (0<=next_pos[0]<COLS and 0<=next_pos[1]<ROWS and next_pos not in obs):
           
           movmnt = random.randint(0,3)
           next_pos = move(movmnt,test_pos)
        kromozom.append(movmnt)
        test_pos = next_pos
    population.append(kromozom)
  population =sort_population_by_fitness_function(population,player_pos,hostage_pos)
  return population    
        

def Check_Kromozom_Health(player_pos,kromozom,obs):
  curr_pos = player_pos
  for item in kromozom:
    curr_pos =  move(item,curr_pos)
    if not (0<=curr_pos[0]<COLS and 0<=curr_pos[1]<ROWS and curr_pos not in obs):
        return False
  return True


# Crossover function
def crossover(parent1, parent2,player_pos,obs):
       global kromozom_size,generations,population_size
    

       child = [0 for i in range(kromozom_size)]
       status = 1
       test_pos = player_pos
       for i in range(kromozom_size):
            num1 = parent1[i]
            num2 = parent2[i]
            new1=move(num1,test_pos)
            new2=move(num2,test_pos)
            if not(0<=new1[0]<COLS and 0<=new1[1]<ROWS and new1 not in obs):
                new1=None
            if not(0<=new2[0]<COLS and 0<=new2[1]<ROWS and new2 not in obs):
                new2 = None
            if new1!=None and new2!=None:
                dist1 = math.dist(new1,hostage_pos)
                dist2 = math.dist(new2,hostage_pos)
                if dist1<dist2:
                    child[i]=num1
                    test_pos = new1
                else:
                    child[i]=num2
                    test_pos=new2
            elif num1==None and num2 != None:
                child[i] = num2
                test_pos=new2
            elif num1!=None and num2==None:
                child[i]=num1
                test_pos=new1
            else:
                counter=20
                status=0
                while(counter>=0):
                    counter-=1
                    rand = random.randint(0,3)
                    newpos=move(rand,test_pos)
                    if (0<=newpos[0]<COLS and 0<=newpos[1]<ROWS and newpos not in obs):
                       child[i]=rand
                       test_pos=newpos
                       status=1
                       break

            
       if status:
           return child
       else :
           return []

# Mutation function
def mutate(individual):
   
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:  # Apply mutation based on the mutation rate
            individual[i] = random.randint(0, 3)  # Randomly mutate to a new direction (0, 1, 2, or 3)
    return individual

def sort_population_by_fitness_function(population,playes_pos,hostage_pos):
   return sorted(population, key=lambda item:fitness(item,player_pos,hostage_pos))

# Function for Genetic Algorithm
def genetic_algorithm(player_pos, hostage_pos, obstacles,population):
    global population_size,generations,kromozom_size
    # print(generations)
    generations-=1
    if generations<0:
        return [-1]
    else:

        sorted_pop = sort_population_by_fitness_function(population,player_pos,hostage_pos)
        mi_flag=0
        new_population = []
        for i in range(0,len(sorted_pop)):
            for j in range(i+1,len(sorted_pop)):
                par1 = sorted_pop[i]
                par2 = sorted_pop[j]
                child = crossover(par1,par2,player_pos,obstacles)
                if len(child)==0 :
                    ...
                else:
                    child = mutate(child)
                    if Check_Kromozom_Health(player_pos,child,obstacles) :
                      new_population.append(child)
                if len(new_population)==population_size:
                   mi_flag=1
                   break
            if mi_flag:
                break
            
            

        return new_population
        
       
    

    
    




##############################################################################################################


#Objective: Check if the player is stuck in a repeating loop.
def in_loop(recent_positions, player):
    # i will change this part if it was false

    return True if recent_positions.count(player)>3 else False

def in_loop_gen(recent_positions, player):
    # i will change this part if it was false

    return True if recent_positions.count(player)>(kromozom_size//2) else False

#Objective: Make a random safe move to escape loops or being stuck.
def random_move(player, obstacles):

    neighbors = [[-1,0],[0,-1],[1,0],[0,1]]
    choices = []
    for item in neighbors:
       item[0]+=player_pos[0]
       item[1]+=player_pos[1]

       if item[0]>=0 and item[0]<COLS and item[1]>=0 and item[1]<ROWS :
           if item not in obstacles:
               choices.append(item)
        
    if choices:
        return random.choice(choices)
    else:
        return player


#Objective: Update the list of recent positions. 
def store_recent_position(recent_positions, new_player_pos, max_positions=MAX_RECENT_POSITIONS):
    if len(recent_positions)==max_positions:
        recent_positions.pop(0)
    recent_positions.append(new_player_pos)

def store_recent_position_gen(recent_positions, new_player_pos, max_positions=5*kromozom_size):
    if len(recent_positions)==max_positions:
        recent_positions.pop(0)
    recent_positions.append(new_player_pos)

# Function to show victory flash
def victory_flash():
    for _ in range(5):
        screen.fill(FLASH_COLOR)
        pygame.display.flip()
        pygame.time.delay(100)
        screen.fill(WHITE)
        pygame.display.flip()
        pygame.time.delay(100)

# Function to show a button and wait for player's input
def show_button_and_wait(message, button_rect):
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, BUTTON_TEXT_COLOR)
    button_rect.width = text.get_width() + 20
    button_rect.height = text.get_height() + 10
    button_rect.center = (WIDTH // 2, HEIGHT // 2)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    screen.blit(text, (button_rect.x + (button_rect.width - text.get_width()) // 2,
                       button_rect.y + (button_rect.height - text.get_height()) // 2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False

# Function to get the algorithm choice from the player
def get_algorithm_choice():
    print("Choose an algorithm:")
    print("1: Hill Climbing")
    print("2: Simulated Annealing")
    print("3: Genetic Algorithm")

    while True:
        choice = input("Enter the number of the algorithm you want to use (1/2/3): ")
        if choice == "1":
            return hill_climbing
        elif choice == "2":
            return simulated_annealing
        elif choice == "3":
            return genetic_algorithm
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")

break_flag= False

def Show_Move(player_pos,obstacles,hostage_pos):
          screen.fill(WHITE)
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
          for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, LIGHT_GREY, rect, 1)

        # Draw obstacles
          for idx, obs in enumerate(obstacles):
                obs_rect = pygame.Rect(obs[0] * TILE_SIZE, obs[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                screen.blit(obstacle_images[idx], obs_rect)

            # Draw player
          player_rect = pygame.Rect(player_pos[0] * TILE_SIZE, player_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
          screen.blit(player_image, player_rect)

            # Draw hostage
          hostage_rect = pygame.Rect(hostage_pos[0] * TILE_SIZE, hostage_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
          screen.blit(hostage_image, hostage_rect)

            # Check if player reached the hostage
          if player_pos[0] == hostage_pos[0] and player_pos[1]==hostage_pos[1]:
                print("Hostage Rescued!")
                victory_flash()  # Show the victory flash
                generations = 100
                population = []
                player_pos = []
                obstacles=  []
                prev_best_kromo=  None
                break_flag = True
                show_button_and_wait("New Game", button_rect)
                start_new_game()
                while len(population)==0:
                    print("Random Generation Created")
                    player_pos= random_move(player_pos,obstacles)
                    Show_Move(player_pos,obstacles,hostage_pos)
                    population = generate_population(player_pos,obstacles,hostage_pos)
                generations-=1
          pygame.display.flip()
          clock.tick(5)



# Main game loop
running = True
clock = pygame.time.Clock()
start_new_game()
button_rect = pygame.Rect(0, 0, 0, 0)
population = []
prev_best_kromo = None
# Get the algorithm choice from the player
chosen_algorithm = get_algorithm_choice()

if chosen_algorithm==genetic_algorithm:
    while len(population)==0:
           print("Random Generation Created")
           player_pos= random_move(player_pos,obstacles)
           Show_Move(player_pos,obstacles,hostage_pos)
           population = generate_population(player_pos,obstacles,hostage_pos)
    generations-=1




while running:
   
    if chosen_algorithm is not genetic_algorithm:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Perform the chosen algorithm step
        new_player_pos = chosen_algorithm(player_pos, hostage_pos, obstacles)
        if new_player_pos[0]==-1 and new_player_pos[1]==-1:
            print("Hostage Can't Be Saved Using ",chosen_algorithm.__name__)
            temperature = 100
            show_button_and_wait("New Game", button_rect)
            start_new_game()
            pygame.display.flip()
            clock.tick(5)  
            continue

        # Check for stuck situations
        if new_player_pos == player_pos or in_loop(recent_positions, new_player_pos):
            # Perform a random move when stuck
            new_player_pos = random_move(player_pos, obstacles)

        # # Update recent positions
        store_recent_position(recent_positions, new_player_pos)
        # Update player's position
        player_pos = new_player_pos

        # Draw the grid background
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, LIGHT_GREY, rect, 1)

        # Draw obstacles
        for idx, obs in enumerate(obstacles):
            obs_rect = pygame.Rect(obs[0] * TILE_SIZE, obs[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            screen.blit(obstacle_images[idx], obs_rect)

        # Draw player
        player_rect = pygame.Rect(player_pos[0] * TILE_SIZE, player_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(player_image, player_rect)

        # Draw hostage
        hostage_rect = pygame.Rect(hostage_pos[0] * TILE_SIZE, hostage_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(hostage_image, hostage_rect)

        # Check if player reached the hostage
        if player_pos[0] == hostage_pos[0] and player_pos[1]==hostage_pos[1]:
            print("Hostage Rescued!")
            victory_flash()  # Show the victory flash
            temperature = 100
            show_button_and_wait("New Game", button_rect)
            start_new_game()
        pygame.display.flip()
        clock.tick(5)  

  
    else:
       
       
       new_population =  genetic_algorithm(player_pos,hostage_pos,obstacles,population)


       if len(new_population)==1 and new_population[0]==-1 or generations<0:
                print("Hostage Can't Be Saved Using Genetic Alg")
                generations = 100
                population = []
                player_pos = []
                obstacles=  []
                prev_best_kromo=  None
                break_flag = True
                show_button_and_wait("New Game", button_rect)
                start_new_game()
                pygame.display.flip()
                clock.tick(5) 
                continue
     
       while len(new_population)==0:
           print("Random Generation Created2")
           player_pos= random_move(player_pos,obstacles)
           Show_Move(player_pos,obstacles,hostage_pos)
           new_population = generate_population(player_pos,obstacles,hostage_pos)
        #    generations-=1
       
       population =sort_population_by_fitness_function(new_population,player_pos,hostage_pos)
       
       best_kromo = population[0]
       if best_kromo==prev_best_kromo:
           player_pos=random_move(player_pos,obstacles)
           prev_best_kromo = None
           continue
       prev_best_kromo = best_kromo
       print("Generation num : ",generations)
       print(best_kromo)
       for num in best_kromo:
           new_pos= move(num,player_pos)
           hostage_pos2 = hostage_pos
           print(num,end=' ')
           player_pos = new_pos
           store_recent_position_gen(recent_positions_gen, new_pos)
           Show_Move(player_pos,obstacles,hostage_pos)
           if new_pos==hostage_pos2:
                generations = 100
                population = []
                # player_pos = []
                # obstacles=  []
                prev_best_kromo=  None
                break_flag = True
                break
           if in_loop_gen(recent_positions_gen,player_pos):
               player_pos =random_move(player_pos,obstacles)
               population=[]
               break
       print('\n')                     
        

        
       
       
           


    

          
    



pygame.quit()
