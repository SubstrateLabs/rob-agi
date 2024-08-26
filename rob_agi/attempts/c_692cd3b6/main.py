from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_692cd3b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two 'C' shapes with yellow, extending to grid edges when necessary.
    
    1. Identify the two 'C' shapes (red color 2 with gray color 5 inside).
    2. Perform a flood fill starting from one 'C' shape to connect to the other.
    3. Extend yellow to grid edges if a 'C' shape touches an edge.
    4. Clean up unnecessary yellow areas.
    5. Preserve the original 'C' shapes.
    
    This approach ensures the minimum necessary yellow area to connect the 'C' shapes
    while extending to the grid edges when required.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify 'C' shapes
    c_shapes = find_c_shapes(input_grid)
    
    # Step 2: Create a new grid and perform flood fill
    new_grid = input_grid.deep_copy()
    flood_fill(new_grid, c_shapes)
    
    # Step 3: Extend yellow to grid edges if necessary
    extend_to_edges(new_grid, c_shapes)
    
    # Step 4: Clean up unnecessary yellow
    clean_up_yellow(new_grid)
    
    # Step 5: Preserve original 'C' shapes
    preserve_c_shapes(new_grid, c_shapes)
    
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

def flood_fill(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    rows, cols = grid.get_dimensions()
    start = find_start_point(grid, c_shapes[0])
    queue = [start]
    visited = set()
    
    while queue:
        r, c = queue.pop(0)
        if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols):
            continue
        visited.add((r, c))
        
        if grid.values[r][c] == 0:
            grid.values[r][c] = 4
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                queue.append((r + dr, c + dc))

def find_start_point(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    for r, c in shape:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 0:
                return (nr, nc)
    return shape[0]  # Fallback to first point in shape if no adjacent black cell

def extend_to_edges(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    rows, cols = grid.get_dimensions()
    for shape in c_shapes:
        for r, c in shape:
            if r == 0 or r == rows - 1:
                for i in range(cols):
                    if grid.values[r][i] == 0:
                        grid.values[r][i] = 4
            if c == 0 or c == cols - 1:
                for i in range(rows):
                    if grid.values[i][c] == 0:
                        grid.values[i][c] = 4

def clean_up_yellow(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    def clean_edge(edge_cells):
        for r, c in edge_cells:
            if grid.values[r][c] == 4:
                grid.values[r][c] = 0
            else:
                break

    # Clean top and bottom edges
    for c in range(cols):
        clean_edge([(r, c) for r in range(rows)])
        clean_edge([(r, c) for r in range(rows-1, -1, -1)])
    
    # Clean left and right edges
    for r in range(rows):
        clean_edge([(r, c) for c in range(cols)])
        clean_edge([(r, c) for c in range(cols-1, -1, -1)])

def preserve_c_shapes(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        for r, c in shape:
            if grid.values[r][c] == 4:
                grid.values[r][c] = 2
