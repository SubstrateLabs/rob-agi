from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_692cd3b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by filling the space between two 'C' shapes with yellow.
    
    1. Identify the two 'C' shapes (red color 2 with gray color 5 inside).
    2. Fill the entire grid with yellow (color 4).
    3. Copy the original 'C' shapes back to their positions.
    4. Clear the rows and columns outside the 'C' shapes.
    
    This approach ensures the yellow area connects the two 'C' shapes while
    preserving their original positions and the grid's structure.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify 'C' shapes
    c_shapes = find_c_shapes(input_grid)
    
    # Step 2: Create a new grid filled with yellow
    new_grid = ColoredGrid(values=[[4 for _ in range(cols)] for _ in range(rows)])
    
    # Step 3: Copy original 'C' shapes
    for shape in c_shapes:
        for r, c in shape:
            new_grid.values[r][c] = input_grid.values[r][c]
    
    # Step 4: Clear rows and columns outside 'C' shapes
    occupied_rows, occupied_cols = get_occupied_rows_cols(c_shapes)
    clear_outside_areas(new_grid, occupied_rows, occupied_cols)
    
    return new_grid

def find_c_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
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

def get_occupied_rows_cols(shapes: List[List[Tuple[int, int]]]) -> Tuple[Set[int], Set[int]]:
    occupied_rows = set()
    occupied_cols = set()
    for shape in shapes:
        for r, c in shape:
            occupied_rows.add(r)
            occupied_cols.add(c)
    return occupied_rows, occupied_cols

def clear_outside_areas(grid: ColoredGrid, occupied_rows: Set[int], occupied_cols: Set[int]):
    rows, cols = grid.get_dimensions()
    
    for r in occupied_rows:
        left_bound = min(c for c in range(cols) if grid.values[r][c] == 2)
        right_bound = max(c for c in range(cols) if grid.values[r][c] == 2)
        for c in range(cols):
            if c < left_bound or c > right_bound:
                grid.values[r][c] = 0
    
    for c in occupied_cols:
        top_bound = min(r for r in range(rows) if grid.values[r][c] == 2)
        bottom_bound = max(r for r in range(rows) if grid.values[r][c] == 2)
        for r in range(rows):
            if r < top_bound or r > bottom_bound:
                grid.values[r][c] = 0
