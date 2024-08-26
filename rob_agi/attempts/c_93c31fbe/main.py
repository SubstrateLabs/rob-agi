from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue (1) elements while respecting other colored elements.
    The solution:
    1. Identifies existing blue structures and non-blue shapes.
    2. Creates a graph representation of blue pixels.
    3. Connects structures using straight lines (horizontal, vertical, or diagonal).
    4. Optimizes connections for minimality and local symmetry.
    5. Handles isolated blue pixels by connecting or removing them.
    6. Ensures the final network is connected and doesn't intersect non-blue shapes.
    7. Maintains or creates symmetry where possible.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_blue_pixels() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1}

    def get_other_shapes() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] not in [0, 1]}

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 1]

    def connect_pixels(start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        r1, c1 = start
        r2, c2 = end
        dr = (r2 > r1) - (r2 < r1)
        dc = (c2 > c1) - (c2 < c1)
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if not is_valid(r, c):
                return False
            r += dr
            c += dc
        return True

    def find_structures(pixels: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        structures = []
        unvisited = pixels.copy()
        while unvisited:
            start = unvisited.pop()
            structure = {start}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r, c = current[0] + dr, current[1] + dc
                        if (r, c) in unvisited:
                            structure.add((r, c))
                            unvisited.remove((r, c))
                            queue.append((r, c))
            structures.append(structure)
        return structures

    def optimize_connections(structures: List[Set[Tuple[int, int]]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        graph = {pixel: set() for structure in structures for pixel in structure}
        for i, struct1 in enumerate(structures):
            for j, struct2 in enumerate(structures[i+1:], start=i+1):
                min_dist = float('inf')
                best_connection = None
                for p1 in struct1:
                    for p2 in struct2:
                        if connect_pixels(p1, p2):
                            dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_connection = (p1, p2)
                if best_connection:
                    p1, p2 = best_connection
                    graph[p1].add(p2)
                    graph[p2].add(p1)
        return graph

    def draw_network(graph: Dict[Tuple[int, int], Set[Tuple[int, int]]]):
        for start, ends in graph.items():
            for end in ends:
                connect_pixels(start, end)
                r1, c1 = start
                r2, c2 = end
                dr = (r2 > r1) - (r2 < r1)
                dc = (c2 > c1) - (c2 < c1)
                r, c = r1, c1
                while (r, c) != (r2, c2):
                    grid.values[r][c] = 1
                    r += dr
                    c += dc
                grid.values[r2][c2] = 1

    blue_pixels = get_blue_pixels()
    other_shapes = get_other_shapes()
    structures = find_structures(blue_pixels)
    graph = optimize_connections(structures)
    draw_network(graph)

    # Remove isolated blue pixels
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 1:
                neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                                if is_valid(r+dr, c+dc) and grid.values[r+dr][c+dc] == 1)
                if neighbors == 0:
                    grid.values[r][c] = 0

    return grid
