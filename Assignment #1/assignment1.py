# Gurleen 
# Fatema 
# Javeera Faizi 101191910
# Name this file to assignment1.py when you submit

import csv
import heapq  # For priority queue implementation

def grid_traversal(filepath):
  # filepath is the path to a CSV file containing a grid 
  grid = []
  with open(filepath, newline='') as file:
    reader = csv.reader(file)
    for row in reader:
      grid.append(row)
  # grid contains all the rows with labels from the csv file (X, O, 1-9, S, G)

  #get all the labels (X, O, 1-9, S, G)
  goals = [] #location of goals
  walls = set() #location of walls
  treasures = {} #stores location & value
  rows = len(grid)
  cols = len(grid[0]) if rows > 0 else 0

  #loop through each coordinate of the grid
  for i in range(rows):
    for j in range(cols):
      if grid[i][j] == 'S':
        start = (i, j)
      elif grid[i][j] == 'G':
        goals.append((i, j))
      elif grid[i][j] == 'X':
        walls.add((i, j))
      elif grid[i][j].isdigit():
        treasures[(i, j)] = int(grid[i][j])

  return grid, start, goals, walls, treasures, rows, cols


def heuristic(a, b):
    #Manhattan distance as heuristic
    #ideal for grids with no diagonals
    #a = [x1, y1], b = [x2, y2]
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):

  grid, start_node, goals, walls, treasures, rows, cols = grid_traversal(filepath)

  # A* search algorithm
  frontier = []
  start_g = 0
  start_h = min(heuristic(start_node, goal) for goal in goals)
  start_f = start_g + start_h
  heapq.heappush(frontier, (start_f, start_g, start_node, 0, [start_node]))  # (f, g, pos, treasure_sum, path)

  explored = set()  # closed set of (pos, treasure_sum)
  num_states_explored = 0

  while frontier:
    f, g, current, t_sum, path = heapq.heappop(frontier)
    num_states_explored += 1

    #check if we are at the goal and have enough treasure
    if current in goals and t_sum >= 5:
      optimal_path = path
      optimal_path_cost = g
      break #return

    #adding our node & treasure sum to the explored nodes.
    state = (current, t_sum)
    if state in explored:
      continue
    explored.add(state)  

    # Explore neighbors (up, down, left, right)
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
      neighbour_row, neighbour_col = current[0] + dx, current[1] + dy
      if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols): #if index out of bounds
        continue
      if (neighbour_row, neighbour_col) in walls: #if wall or obstacle
        continue

      #calculate new g, h, f values
      new_t = t_sum + treasures.get((neighbour_row, neighbour_col), 0)
      new_g = g + 1
      new_h = min(heuristic((neighbour_row, neighbour_col), goal) for goal in goals)
      new_f = new_g + new_h

      #getting minimum f value
      heapq.heappush(frontier, (new_f, new_g, (neighbour_row, neighbour_col), new_t, path + [(neighbour_row, neighbour_col)]))

  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search
  return optimal_path, optimal_path_cost, num_states_explored


result = pathfinding("Assignment #1\Examples\Examples\Example0\grid.txt")
print(result)

# right now code is traversing over the same treasure multiple times 
# general -   how detailed should assignment answers be 
#             is code formatting correct
#             how to submit - one person or multiple
#             is heapq.heappush allowed (it's in standard python library)




# ####################################################

# from heapq import heappush, heappop

# grid = [
#     ["S","0","G"],
#     ["0","0","0"],
#     ["5","X","0"],
# ]

# rows, cols = len(grid), len(grid[0])

# # Parse grid
# start = None
# goals = set()
# walls = set()
# treasures = {}

# for i in range(rows):
#     for j in range(cols):
#         cell = grid[i][j]
#         if cell == "S":
#             start = (i, j)
#         elif cell == "G":
#             goals.add((i, j))
#         elif cell == "X":
#             walls.add((i, j))
#         elif cell.isdigit() and cell != "0":
#             treasures[(i, j)] = int(cell)

# def manhattan(a, b):
#      abs(a[0] - b[0]) + abs(a[1] - b[1])

# # A* with detailed tracing
# frontier = []
# start_g = 0
# start_h = min(manhattan(start, g) for g in goals)
# start_f = start_g + start_h
# heappush(frontier, (start_f, start_g, start, 0, [start]))  # (f, g, pos, treasure_sum, path)

# visited = set()  # closed set of (pos, treasure_sum)
# trace_lines = []
# num_states_explored = 0

# def log(msg):
#     trace_lines.append(msg)

# log("=== STEP-BY-STEP A* TRACE ===")
# log(f"Grid:\n 0: {grid[0]}\n 1: {grid[1]}\n 2: {grid[2]}")
# log(f"Start: {start}, Goals: {goals}, Walls: {walls}, Treasures: {treasures}")
# log("")

# solution = None

# while frontier:
#     f, g, current, t_sum, path = heappop(frontier)
#     num_states_explored += 1

#     # Compute heuristic to nearest goal for logging
#     h_cur = min(manhattan(current, g_) for g_ in goals)
#     log(f"POP  #{num_states_explored}: pos={current}, g={g}, h={h_cur}, f={f}, treasure_sum={t_sum}")

#     # Goal check (must have t_sum >= 5)
#     if current in goals:
#         if t_sum >= 5:
#             log(f"  -> Goal with enough treasure! RETURN path len={len(path)}, cost={g}")
#             solution = (path, g, num_states_explored)
#             break
#         else:
#             log("  -> At a GOAL, but treasure_sum < 5. CONTINUE exploring.")

#     state = (current, t_sum)
#     if state in visited:
#         log("  -> Already visited this (pos, treasure_sum). Skip expansion.")
#         continue
#     visited.add(state)

#     # Explore neighbors
#     for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
#         nr, nc = current[0] + dx, current[1] + dy
#         if not (0 <= nr < rows and 0 <= nc < cols):
#             continue
#         if (nr, nc) in walls:
#             continue

#         new_t = t_sum + treasures.get((nr, nc), 0)
#         new_g = g + 1
#         h = min(manhattan((nr, nc), g_) for g_ in goals)
#         new_f = new_g + h

#         heappush(frontier, (new_f, new_g, (nr, nc), new_t, path + [(nr, nc)]))
#         log(f"    PUSH: to={(nr, nc)}, g={new_g}, h={h}, f={new_f}, treasure_sum={new_t}")

#     log("")

# if solution is None:
#     log("No valid solution found (should not happen in this demo).")

# # Print the trace for the user to read
# print("\n".join(trace_lines))
