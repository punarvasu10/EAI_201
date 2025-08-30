from collections import defaultdict, deque
import heapq
import math

class PipeNetwork:
    def __init__(self):
        self.graph = defaultdict(list)
        self.coords = {}  # store coordinates of junctions

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))  # undirected graph

    def add_coord(self, node, x, y):
        self.coords[node] = (x, y)

    # Depth First Search (DFS)
    def dfs(self, start, goal):
        stack = [(start, [start], 0)]
        visited = set()
        while stack:
            node, path, cost = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                return path, cost, len(visited)
            for neigh, c in self.graph[node]:
                if neigh not in visited:
                    stack.append((neigh, path+[neigh], cost+c))
        return None, float("inf"), len(visited)

    # Breadth First Search (BFS)
    def bfs(self, start, goal):
        queue = deque([(start, [start], 0)])
        visited = {start}
        while queue:
            node, path, cost = queue.popleft()
            if node == goal:
                return path, cost, len(visited)
            for neigh, c in self.graph[node]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, path+[neigh], cost+c))
        return None, float("inf"), len(visited)

    # Uniform Cost Search (UCS)
    def ucs(self, start, goal):
        pq = [(0, start, [start])]
        visited = {}
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal:
                return path, cost, len(visited)
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neigh, c in self.graph[node]:
                heapq.heappush(pq, (cost+c, neigh, path+[neigh]))
        return None, float("inf"), len(visited)

    # A* Search
    def astar(self, start, goal):
        def heuristic(a, b):
            (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        pq = [(heuristic(start, goal), 0, start, [start])]
        visited = {}
        while pq:
            est_total, cost, node, path = heapq.heappop(pq)
            if node == goal:
                return path, cost, len(visited)
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            for neigh, c in self.graph[node]:
                g = cost + c
                f = g + heuristic(neigh, goal)
                heapq.heappush(pq, (f, g, neigh, path+[neigh]))
        return None, float("inf"), len(visited)

# Example usage
if __name__ == "__main__":
    network = PipeNetwork()

    # Pipes: (junction1, junction2, cost)
    pipes = [
        ("a","b",4), ("a","c",2), ("b","c",5), ("b","d",10),
        ("c","e",3), ("e","d",4), ("d","f",11)
    ]
    for u,v,c in pipes:
        network.add_pipe(u,v,c)

    # Coordinates of junctions (for A*)
    coords = {
        "a":(0,0), "b":(2,1), "c":(1,3),
        "d":(5,2), "e":(3,4), "f":(6,3)
    }
    for node,(x,y) in coords.items():
        network.add_coord(node,x,y)

    start, goal = "a", "f"

    print("DFS:", network.dfs(start, goal))
    print("BFS:", network.bfs(start, goal))
    print("UCS:", network.ucs(start, goal))
    print("A* :", network.astar(start, goal))
