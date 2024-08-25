from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_03560426(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging colored shapes into a compact arrangement.
    
    1. Extracts shapes from bottom to top, preserving their order.
    2. Places shapes in a new grid, starting from the top-left corner.
    3. Stacks shapes vertically, creating new stacks to the right when needed.
    4. Maintains the original form and orientation of each shape.
    5. Compacts the arrangement by moving shapes left if possible.
    6. Fills remaining space with black (0).
    
    Returns the transformed grid with shapes arranged compactly in the top-left quadrant.
    """
    shapes = extract_shapes(input_grid)
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    place_shapes(shapes, output_grid)
    compact_arrangement(output_grid)
    return output_grid

def extract_shapes(grid: ColoredGrid) -> List[Dict[str, any]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows-1, -1, -1):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                shape = bfs(grid, r, c, visited)
                shapes.append({"color": grid.values[r][c], "coords": shape})
    
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

def place_shapes(shapes: List[Dict[str, any]], output_grid: ColoredGrid) -> None:
    stacks = [0]  # Keep track of the bottom of each stack
    for shape in shapes:
        placed = False
        for i, bottom in enumerate(stacks):
            if can_place_shape(output_grid, shape["coords"], (bottom, i)):
                place_shape(output_grid, shape["coords"], (bottom, i), shape["color"])
                stacks[i] = bottom + max(y for _, y in shape["coords"]) + 1
                placed = True
                break
        if not placed:
            # Start a new stack
            stacks.append(0)
            place_shape(output_grid, shape["coords"], (0, len(stacks) - 1), shape["color"])
            stacks[-1] = max(y for _, y in shape["coords"]) + 1

def can_place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int]) -> bool:
    rows, cols = grid.get_dimensions()
    for x, y in shape:
        new_x, new_y = position[0] + x, position[1] + y
        if new_x < 0 or new_x >= rows or new_y < 0 or new_y >= cols or grid.values[new_x][new_y] != 0:
            return False
    return True

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int], color: int) -> None:
    for x, y in shape:
        new_x, new_y = position[0] + x, position[1] + y
        grid.values[new_x][new_y] = color

def compact_arrangement(grid: ColoredGrid) -> None:
    rows, cols = grid.get_dimensions()
    for c in range(1, cols):
        for r in range(rows):
            if grid.values[r][c] != 0:
                move_left = 0
                while c - move_left > 0 and all(grid.values[i][c - move_left - 1] == 0 for i in range(rows)):
                    move_left += 1
                if move_left > 0:
                    for i in range(rows):
                        grid.values[i][c - move_left] = grid.values[i][c]
                        grid.values[i][c] = 0
