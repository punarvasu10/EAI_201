import heapq
import math
import time

grid = [
    ['S', 0, 0, 1, 0],
    [1, 1, 0, 1, 'G'],
    [0, 5, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
]

rows, cols = len(grid), len(grid[0])

# Locate Start & Goal
for i in range(rows):
    for j in range(cols):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)
# Heuristics
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def diagonal(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Cost function (ghost zones)
def cost(x, y):
    if grid[x][y] == 5:  
        return 5
    return 1

# Neighbor function (4 or 8 moves)
def neighbors(node, allow_diagonal=True):
    x, y = node
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    if allow_diagonal:
        directions += [(1,1), (1,-1), (-1,1), (-1,-1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
            yield (nx, ny)

# Path reconstruction
def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    return path[::-1]

# Greedy Best-First Search
def greedy_bfs(start, goal, heuristic):
    frontier = [(heuristic(start, goal), start)]
    visited = set([start])
    parent = {}
    explored = 0
    while frontier:
        _, node = heapq.heappop(frontier)
        explored += 1
        if node == goal:
            return reconstruct_path(parent, start, goal), explored
        for nb in neighbors(node, allow_diagonal=True):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                heapq.heappush(frontier, (heuristic(nb, goal), nb))
    return None, explored
# A* Search
def astar(start, goal, heuristic):
    frontier = [(heuristic(start, goal), 0, start)]
    visited = {}
    parent = {}
    explored = 0
    while frontier:
        f, g, node = heapq.heappop(frontier)
        explored += 1
        if node == goal:
            return reconstruct_path(parent, start, goal), explored
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        for nb in neighbors(node, allow_diagonal=True):
            new_g = g + cost(*nb)
            heapq.heappush(frontier, (new_g + heuristic(nb, goal), new_g, nb))
            parent[nb] = node
    return None, explored

# Visualization
def print_path(path):
    temp = [[str(cell) for cell in row] for row in grid]
    if path:
        for (x,y) in path:
            if temp[x][y] not in ('S','G'):
                temp[x][y] = '*'
    for row in temp:
        print(" ".join(row))
    print()

# Run experiments
for algo, func in [("Greedy BFS", greedy_bfs), ("A*", astar)]:
    for hname, hfunc in [("Manhattan", manhattan), ("Euclidean", euclidean), ("Diagonal", diagonal)]:
        start_time = time.time()
        path, explored = func(start, goal, hfunc)
        end_time = time.time()
        if path:
            print(f"{algo} with {hname}:")
            print(f"  Path length = {len(path)}")
            print(f"  Nodes explored = {explored}")
            print(f"  Execution time = {end_time - start_time:.6f} sec")
            print("  Path visualization:")
            print_path(path)
        else:
            print(f"{algo} with {hname}: No path found\n")