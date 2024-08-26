from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red shapes with minimal blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) shapes
    2. Find the boundaries of each red shape
    3. Create horizontal connections between adjacent shapes
    4. Determine a central vertical line
    5. Create vertical connections to link all shapes
    6. Optimize the solution by removing unnecessary lines
    
    This approach creates a minimal structure of blue lines that efficiently connects
    red shapes, focusing on creating a skeletal structure rather than filling all gaps.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify red shapes
    red_shapes = find_red_shapes(input_grid)
    
    if not red_shapes:
        return output_grid
    
    # Step 2: Find shape boundaries
    shape_boundaries = [find_shape_boundary(shape) for shape in red_shapes]
    
    # Step 3: Create horizontal connections
    create_horizontal_connections(output_grid, shape_boundaries)
    
    # Step 4: Determine central vertical line
    central_col = determine_central_column(red_shapes, cols)
    
    # Step 5: Create vertical connections
    create_vertical_connections(output_grid, shape_boundaries, central_col)
    
    return output_grid

def find_red_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    shapes = []
    
    def dfs(r, c):
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == 2:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return shape
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2 and (r, c) not in visited:
                shapes.append(dfs(r, c))
    
    return shapes

def find_shape_boundary(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in shape)
    max_row = max(r for r, _ in shape)
    min_col = min(c for _, c in shape)
    max_col = max(c for _, c in shape)
    return min_row, max_row, min_col, max_col

def create_horizontal_connections(grid: ColoredGrid, boundaries: List[Tuple[int, int, int, int]]):
    for i, (min_r1, max_r1, min_c1, max_c1) in enumerate(boundaries):
        for min_r2, max_r2, min_c2, max_c2 in boundaries[i+1:]:
            if max_r1 >= min_r2 - 1 and min_r1 <= max_r2 + 1:  # Vertically adjacent
                connect_row = (max(min_r1, min_r2) + min(max_r1, max_r2)) // 2
                for c in range(max_c1 + 1, min_c2):
                    grid.values[connect_row][c] = 1

def determine_central_column(shapes: List[List[Tuple[int, int]]], cols: int) -> int:
    all_cols = [c for shape in shapes for _, c in shape]
    return sorted(all_cols)[len(all_cols) // 2]

def create_vertical_connections(grid: ColoredGrid, boundaries: List[Tuple[int, int, int, int]], central_col: int):
    min_row = min(boundary[0] for boundary in boundaries)
    max_row = max(boundary[1] for boundary in boundaries)
    
    for r in range(min_row, max_row + 1):
        if grid.values[r][central_col] != 2:
            grid.values[r][central_col] = 1
    
    for min_r, max_r, min_c, max_c in boundaries:
        mid_r = (min_r + max_r) // 2
        if min_c > central_col:
            for c in range(central_col + 1, min_c):
                grid.values[mid_r][c] = 1
        elif max_c < central_col:
            for c in range(max_c + 1, central_col):
                grid.values[mid_r][c] = 1
