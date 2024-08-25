from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_03560426(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging colored shapes into a compact arrangement.
    
    1. Extracts shapes from bottom to top, preserving their order.
    2. Generates all possible rotations for each shape.
    3. Places shapes in a new grid, starting from the top-left corner.
    4. Tries all rotations to find the most compact arrangement.
    5. Maintains the original order of shapes.
    6. Compacts the arrangement by moving shapes left if possible.
    7. Fills remaining space with black (0).
    
    Returns the transformed grid with shapes arranged compactly in the top-left quadrant.
    """
    shapes = extract_shapes(input_grid)
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    place_shapes_with_rotations(shapes, output_grid)
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

def rotate_shape(shape: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    rotations = [shape]
    for _ in range(3):  # Generate 3 more rotations
        new_rotation = [(y, -x) for x, y in rotations[-1]]
        min_x = min(x for x, _ in new_rotation)
        min_y = min(y for _, y in new_rotation)
        new_rotation = [(x - min_x, y - min_y) for x, y in new_rotation]
        rotations.append(new_rotation)
    return rotations

def place_shapes_with_rotations(shapes: List[Dict[str, any]], output_grid: ColoredGrid) -> None:
    for shape in shapes:
        rotations = rotate_shape(shape["coords"])
        best_position = None
        best_rotation = None
        min_area = float('inf')
        
        for rotation in rotations:
            for r in range(10):
                for c in range(10):
                    if can_place_shape(output_grid, rotation, (r, c)):
                        area = calculate_area(output_grid, rotation, (r, c))
                        if area < min_area:
                            min_area = area
                            best_position = (r, c)
                            best_rotation = rotation
        
        if best_position:
            place_shape(output_grid, best_rotation, best_position, shape["color"])

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

def calculate_area(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int]) -> int:
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for x, y in shape:
        new_x, new_y = position[0] + x, position[1] + y
        min_x = min(min_x, new_x)
        max_x = max(max_x, new_x)
        min_y = min(min_y, new_y)
        max_y = max(max_y, new_y)
    return (max_x - min_x + 1) * (max_y - min_y + 1)

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
