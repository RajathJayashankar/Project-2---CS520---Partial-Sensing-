import colors as Color
import heurisitcs as Heuristic
import states as State
from random import randint
import queue as Queue
from matplotlib import pyplot as plt
import timeit
import CSP


# Defines Each Node of the Grid
class Node:
	def __init__(self, row, col):
        # Row and Column of the Node for positon
		self.row = row
		self.col = col

        #State of the node, to assign as barrier
		self.state = State.Unexplored

        #Maintains the neighbours
		self.neighbors = []

    # Returns Positionof the Node
	def get_pos(self):
		return self.row, self.col

    # Returns if Barrier
	def is_barrier(self):
		return self.state == State.Barrier

    # Makes a Node a Barrier
	def make_barrier(self):
		self.state = State.Barrier

    #Updates all neighbours of a Node
	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row + 1][self.col])

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
			self.neighbors.append(grid[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
			self.neighbors.append(grid[self.row][self.col + 1])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
			self.neighbors.append(grid[self.row][self.col - 1])

    #Updates all neighbours of a Node and Updates if Barrier by checking with Original Gridworld
	def update_neighbors_barriers(self, grid_original):
		self.neighbours = []
		if self.row < self.total_rows - 1 and not grid_original[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid_original[self.row + 1][self.col])

		if self.row > 0 and not grid_original[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid_original[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid_original[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid_original[self.row][self.col + 1])

		if self.col > 0 and not grid_original[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid_original[self.row][self.col - 1])


Node_type = {"isStart" : 1,"isEnd" : 2,"isBarrier" : 3,"isNotBarrier" : 4, "isOpen": 5}
Confirmed_type = {"notConfirmed" : 1,"ConfirmedEmpty" : 2,"ConfirmedBlocked" : 3}


class Node_Sensing:
  def __init__(self, row, col, total_rows):
    self.row = row
    self.col = col
    self.node_type = Node_type["isOpen"]
    self.neighbors = []
    self.total_rows = total_rows
    
    self.inferred = False
    self.confirmed = Confirmed_type["notConfirmed"]
    self.Nx = 0
    self.Cx = 0
    self.Bx = 0
    self.Ex = 0
    self.Hx = 8

  def get_pos(self):
    return self.row, self.col

  def is_barrier(self):
    return self.node_type == Node_type['isBarrier']

  def is_start(self):
    return self.node_type == Node_type['isStart']

  def is_end(self):
    return self.node_type == Node_type['isEnd']

  def make_inferred(self):
    self.inferred == True

  def make_start(self):
    self.node_type = Node_type['isStart']

  def make_barrier(self):
    self.node_type = Node_type['isBarrier']

  def make_end(self):
    self.node_type = Node_type['isEnd']

  def update_neighbors(self, grid):
    self.neighbors = []
    if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
      self.neighbors.append(grid[self.row + 1][self.col])

    if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
      self.neighbors.append(grid[self.row - 1][self.col])

    if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
      self.neighbors.append(grid[self.row][self.col + 1])

    if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
      self.neighbors.append(grid[self.row][self.col - 1])

  def update_neighbors_barriers(self, grid, grid_original):
    if self.row < self.total_rows - 1 and grid_original[self.row + 1][self.col].is_barrier(): # DOWN
      grid[self.row + 1][self.col].make_barrier()

    if self.row > 0 and grid_original[self.row - 1][self.col].is_barrier(): # UP
      grid[self.row - 1][self.col].make_barrier()

    if self.col < self.total_rows - 1 and grid_original[self.row][self.col + 1].is_barrier(): # RIGHT
      grid[self.row][self.col + 1].make_barrier()

    if self.col > 0 and grid_original[self.row][self.col - 1].is_barrier(): # LEFT
      grid[self.row][self.col - 1].make_barrier()

  def sense_Neighbours(self, grid, gird_original):
    rowin, colin = self.get_pos()
    nb = 0 # No. of blocked nodes -- Cx
    ncb = 0 # No. of confirmed blocked nodes -- Bx
    nce = 0 # No. of confirmed empty  nodes -- Ex
    nnc = 0 # No. of unconfirmed  nodes -- Hx
    for col in range(colin-1, colin+2):
          for row in range(rowin-1, rowin+2):
              if (-1 < rowin < self.total_rows and 
                  -1 < colin < self.total_rows and 
                  (rowin != row or colin != col) and
                  (0 <= col < self.total_rows) and
                  (0 <= row < self.total_rows)):
                if(gird_original[row][col].is_barrier()):
                  nb += 1
                if(grid[row][col].confirmed == Confirmed_type["notConfirmed"]):
                  nnc += 1
                elif(grid[row][col].confirmed == Confirmed_type["ConfirmedEmpty"]):
                  nce += 1
                elif(grid[row][col].confirmed == Confirmed_type["ConfirmedBlocked"]):
                  ncb += 1
    return nb, nnc, nce, ncb

  def make_Confirmed_Empty(self):
    self.inferred = True
    self.confirmed = Confirmed_type["ConfirmedEmpty"]
  
  def make_Confirmed_Blocked(self):
    self.inferred = True
    self.confirmed = Confirmed_type["ConfirmedBlocked"]

  def update_HiddenNeighbours_Empty(self, grid, grid_original):
    rowin, colin = self.get_pos()
    for col in range(colin-1, colin+2):
          for row in range(rowin-1, rowin+2):
              if (-1 < rowin < self.total_rows and 
                  -1 < colin < self.total_rows and 
                  (rowin != row or colin != col) and
                  (0 <= col < self.total_rows) and
                  (0 <= row < self.total_rows)):
                if(grid[row][col].inferred == False):
                  grid[row][col].inferred == True
                  grid[row][col].confirmed = Confirmed_type["ConfirmedEmpty"]

  def update_HiddenNeighbours_Blocked(self, grid, grid_original):
    rowin, colin = self.get_pos()
    for col in range(colin-1, colin+2):
          for row in range(rowin-1, rowin+2):
              if (-1 < rowin < self.total_rows and 
                  -1 < colin < self.total_rows and 
                  (rowin != row or colin != col) and
                  (0 <= col < self.total_rows) and
                  (0 <= row < self.total_rows)):
                if(grid[row][col].inferred == False):
                  # print("Callig Blocked")
                  grid[row][col].inferred == True
                  grid[row][col].make_barrier()
                  grid[row][col].confirmed = Confirmed_type["ConfirmedBlocked"]

  def findNeighbors(self):
    nn = 0
    rowin, colin = self.get_pos()
    for col in range(colin-1, colin+2):
        for row in range(rowin-1, rowin+2):
            if (-1 < rowin < self.total_rows and 
                -1 < colin < self.total_rows and 
                (rowin != row or colin != col) and
                (0 <= col < self.total_rows) and
                (0 <= row < self.total_rows)):
                nn += 1
    return nn

  def Inference_Agent_3(self, grid, grid_original):
    self.inferred = True
    if(self.Nx == 0):
      self.Nx = self.findNeighbors()
    if(self.Hx != 0):
      self.Cx, self.Hx, self.Ex, self.Bx = self.sense_Neighbours(grid, grid_original)
      # print("-----CX HX EX BX NX NX-CX -----")
      # print(self.Cx, self.Hx, self.Ex, self.Bx, self.Nx, self.Nx - self.Cx)
    else:
      if(self.confirmed == Confirmed_type["ConfirmedEmpty"]):
        return False
      elif(self.confirmed == Confirmed_type["ConfirmedBlocked"]):
        return True
      else:
        return False
    if(self.Cx == self.Bx):
      self.update_HiddenNeighbours_Empty(grid, grid_original)
    if((self.Nx - self.Cx) == self.Ex):
      self.update_HiddenNeighbours_Blocked(grid, grid_original)
    if(self.confirmed == Confirmed_type["ConfirmedEmpty"]):
      return False
    elif(self.confirmed == Confirmed_type["ConfirmedBlocked"]):
      return True
    else:
      return False

  def __lt__(self, other):
    return False


def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		# draw()


def AStarAlgo(draw, grid, start, end, heuristic_func):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {node: float("inf") for row in grid for node in row}
	g_score[start] = 0
	f_score = {node: float("inf") for row in grid for node in row}
	f_score[start] = Heuristic.heuristic_functions[heuristic_func](start.get_pos(), end.get_pos())

	open_set_hash = {start}

	while not open_set.empty():

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + Heuristic.heuristic_functions[heuristic_func](neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

	return False


def make_grid(rows, ):
	grid = []
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			node = Node_Sensing(i, j, rows)
			grid[i].append(node)

	return grid


def make_grid_density(rows, probability):
	grid = []
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			node = Node_Sensing(i, j, rows)
			n = randint(0,100)
			if n < probability: node.make_barrier()
			grid[i].append(node)	

	return grid

def getLabel(i):
	if i == 0:
		return "Manhattan Distance"
	elif i == 1:
		return "Euclidean Distance"
	elif i == 2:
		return "Chebyshev Distance"

def reconstruct_path_repeated_with_Inference(grid, came_from, current, grid_orignal, came_from_2, trajectory_length, no_cells_visited, AgentFlag):
  path = []
  path.insert(0, current)
  while current in came_from:
    current = came_from[current]
    path.insert(0, current)

  unblocked_path = []

  for k in range(len(path)):
    if(AgentFlag == 3):
        path[k].Inference_Agent_3(grid, grid_orignal)
    else:
        grid = CSP.Constraint_Solver(grid)
    trajectory_length += 1
    if(grid_orignal[path[k].row][path[k].col].is_barrier() == False):
      
      path[k].make_Confirmed_Empty()
      path[k].make_inferred()
      unblocked_path.append(path[k])
    
      if (path[k] in came_from_2):
        indexSplice = came_from_2.index(path[k])
        came_from_2 = came_from_2[:indexSplice + 1]
      else:
        came_from_2.append(path[k])  
    else:
      trajectory_length += 1
      path[k].make_barrier()
      path[k].make_Confirmed_Blocked()
      path[k].make_inferred()
      return unblocked_path, came_from_2, trajectory_length, no_cells_visited 

      
  return unblocked_path, came_from_2, trajectory_length, no_cells_visited

def reconstruct_path_repeated(came_from, current, grid_full , came_from_2, trajectory_length, no_cells_visited, direction_flag):
	path = []
	path.insert(0, current)
    # Make the path ffrom start to end that was traversed
	while current in came_from:
		current = came_from[current]
		path.insert(0, current)

	unblocked_path = []
    # Traverse nodes to find if blocked by check with full grid
	for k in range(len(path)):
		trajectory_length += 1 # Update Trajectory Lenght
		if(grid_full [path[k].row][path[k].col].is_barrier() == False):
			if(direction_flag == 2): # Flag to decide between unidirectional and normal Repeated A* 
				path[k].update_neighbors_barriers( grid_full )
			unblocked_path.append(path[k]) 
			if (path[k] in came_from_2):
				indexSplice = came_from_2.index(path[k]) #Update the path it came from
				came_from_2 = came_from_2[:indexSplice + 1]
			else:
				came_from_2.append(path[k])
		else:
			path[k].make_barrier()
			return unblocked_path, came_from_2, trajectory_length, no_cells_visited
	return unblocked_path, came_from_2, trajectory_length, no_cells_visited

def RepeatedAStar(grid, start, end, grid_full, came_from_2, trajectory_length, no_cells_visited, heuristic_used = 1, direction_flag = 1):	
    fringe = Queue.PriorityQueue() 
    fringe.put((0, 0, start)) # Insert into PQ with inital cost as 0
    came_from = {} # Maintain a Dict to find where node came from
    
    # Initialize intial cost of traveral as infinity 
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = Heuristic.Manhattan_Distance(start.get_pos(), end.get_pos())

    fringe_hash = {start}

    while not fringe.empty():

        current = fringe.get()[2] #Return first Node in PQ
        fringe_hash.remove(current)
        no_cells_visited += 1

        if current == end:	#If End node found, return to path reconstruction funtion	
            return reconstruct_path_repeated(came_from, end, draw, grid_full, came_from_2, trajectory_length, no_cells_visited, direction_flag)
        
        current.update_neighbors(grid)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1	 # Update G score by 1 as it moved 1 node away

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current				
                g_score[neighbor] = temp_g_score			
                f_score[neighbor] = temp_g_score + Heuristic.Manhattan_Distance (neighbor.get_pos(), end.get_pos())

                if neighbor not in fringe_hash: #If not in fringe hash add to fringe
                    count += 1
                    fringe.put((f_score[neighbor], count, neighbor))
                    fringe_hash.add(neighbor)

    return False, came_from_2, trajectory_length, no_cells_visited


def RepeatedBFS(draw, grid, start, end, grid_full, came_from_2, trajectory_length, no_cells_visited):
	count = 0
	fringe = Queue()
	fringe.put(start)
	came_from = {}

	g_score = {node: float("inf") for row in grid for node in row}
	g_score[start] = 0
	f_score = {node: float("inf") for row in grid for node in row}
	f_score[start] = Heuristic.Manhattan_Distance(start.get_pos(), end.get_pos())

	fringe_hash = {start}

	while not fringe.empty():
		current = fringe.get()
		fringe_hash.remove(current)
		no_cells_visited += 1

		if current == end:		
			return reconstruct_path_repeated(came_from, end, draw, grid_full, came_from_2, trajectory_length, no_cells_visited)
		
		current.update_neighbors(grid)
		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1	

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current				
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + Heuristic.Manhattan_Distance (neighbor.get_pos(), end.get_pos())

				if neighbor not in fringe_hash:
					count += 1
					fringe.put(neighbor)
					fringe_hash.add(neighbor)

		if current != start:
			pass

	return False, came_from_2, trajectory_length, no_cells_visited


def main(rows, p, iter): # Rows - No. of rows and coluns, p - density range, iter - no. of iterations to be run
	
	Rows = rows

	for i in range(p, 0, -1): # Runs for a tange of P density values till 0
		for j in range(0, iter, 1): # Sets number of iteration per density value
 
			grid_orignal = make_grid_density(Rows, i) # Generated the original gridworld with i - p density value

			grid = make_grid(Rows) # Generated the gridworld to be explored 
			start = grid[0][0]
			end = grid[rows - 1][rows -1]

			came_from_2 = []
			trajectory_length = 0
			no_cells_visited = 0

			while(True): #Run loop until either fringe return empty list or last element in path traveres is end node
				path_traversed, came_from_2, trajectory_length, no_cells_visited = RepeatedAStar(grid, start, end, grid_orignal, came_from_2, trajectory_length, no_cells_visited) 

				if(path_traversed == [] or path_traversed == False):
					print("No Path")
					break				

				if (path_traversed[-1] == end):
					break

				else:
					start = path_traversed[-1] # Update the start node to last node before the blocked state
					start.make_start()

	
# Call main with the parameter needed
main(WIDTH, P, ITER)