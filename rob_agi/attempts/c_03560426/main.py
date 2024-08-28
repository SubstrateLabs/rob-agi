from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

class Shape:
    def __init__(self, color: int, coords: List[Tuple[int, int]], order: int):
        self.color = color
        self.coords = coords
        self.order = order
        self.area = len(coords)
        self.width = max(x for x, _ in coords) - min(x for x, _ in coords) + 1
        self.height = max(y for _, y in coords) - min(y for _, y in coords) + 1

    def rotate(self):
        self.coords = [(y, -x) for x, y in self.coords]
        min_x = min(x for x, _ in self.coords)
        min_y = min(y for _, y in self.coords)
        self.coords = [(x - min_x, y - min_y) for x, y in self.coords]
        self.width, self.height = self.height, self.width

def solve_03560426(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging colored shapes into a compact arrangement.
    
    1. Extracts shapes from bottom to top, left to right.
    2. Analyzes shapes for area and dimensions.
    3. Sorts shapes by area and original order.
    4. Places shapes in a compact arrangement, allowing rotations and slight overlaps.
    5. Optimizes the arrangement by shifting shapes left and up where possible.
    6. Ensures all shapes are connected.
    7. Fills remaining space with black (0).
    
    Returns the transformed grid with shapes arranged compactly in the top-left quadrant.
    """
    shapes = extract_shapes(input_grid)
    shapes.sort(key=lambda s: (-s.area, s.order))
    output_grid = place_shapes(shapes)
    optimize_arrangement(output_grid)
    ensure_connectivity(output_grid)
    return output_grid

def extract_shapes(grid: ColoredGrid) -> List[Shape]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    order = 0
    
    for r in range(rows-1, -1, -1):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                coords = bfs(grid, r, c, visited)
                shapes.append(Shape(grid.values[r][c], coords, order))
                order += 1
    
    return shapes

def bfs(grid: ColoredGrid, start_r: int, start_c: int, visited: set) -> List[Tuple[int, int]]:
    queue = deque([(start_r, start_c)])
    shape = []
    color = grid.values[start_r][start_c]
    rows, cols = grid.get_dimensions()
    min_r, min_c = start_r, start_c
    
    while queue:
        r, c = queue.popleft()
        if (r, c) not in visited and grid.values[r][c] == color:
            visited.add((r, c))
            min_r = min(min_r, r)
            min_c = min(min_c, c)
            shape.append((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))
    
    return [(r - min_r, c - min_c) for r, c in shape]  # Store relative coordinates

def place_shapes(shapes: List[Shape]) -> ColoredGrid:
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for shape in shapes:
        best_score = float('inf')
        best_placement = None
        for rotated in [False, True]:
            if rotated:
                shape.rotate()
            for r in range(10):
                for c in range(10):
                    if can_place_shape(output_grid, shape, r, c):
                        score = r + c + (1 if rotated else 0)
                        if score < best_score:
                            best_score = score
                            best_placement = (r, c, rotated)
            if rotated:
                shape.rotate()  # Rotate back
        
        if best_placement:
            r, c, rotated = best_placement
            if rotated:
                shape.rotate()
            place_shape(output_grid, shape, r, c)
    
    return output_grid

def can_place_shape(grid: ColoredGrid, shape: Shape, r: int, c: int) -> bool:
    overlap = 0
    for x, y in shape.coords:
        nr, nc = r + y, c + x
        if nr < 0 or nr >= 10 or nc < 0 or nc >= 10:
            return False
        if grid.values[nr][nc] != 0:
            overlap += 1
            if overlap > 1:
                return False
    return True

def place_shape(grid: ColoredGrid, shape: Shape, r: int, c: int) -> None:
    for x, y in shape.coords:
        nr, nc = r + y, c + x
        grid.values[nr][nc] = shape.color

def optimize_arrangement(grid: ColoredGrid) -> None:
    changed = True
    while changed:
        changed = False
        for r in range(9, -1, -1):
            for c in range(10):
                if grid.values[r][c] != 0:
                    color = grid.values[r][c]
                    if r > 0 and grid.values[r-1][c] == 0:
                        grid.values[r-1][c] = color
                        grid.values[r][c] = 0
                        changed = True
                    elif c > 0 and grid.values[r][c-1] == 0:
                        grid.values[r][c-1] = color
                        grid.values[r][c] = 0
                        changed = True

def ensure_connectivity(grid: ColoredGrid) -> None:
    visited = set()
    start = next((r, c) for r in range(10) for c in range(10) if grid.values[r][c] != 0)
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and grid.values[r][c] != 0:
            visited.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    stack.append((nr, nc))
    
    for r in range(10):
        for c in range(10):
            if grid.values[r][c] != 0 and (r, c) not in visited:
                grid.values[r][c] = 0
