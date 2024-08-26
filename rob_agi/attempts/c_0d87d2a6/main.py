from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_0d87d2a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue dots with a path and filling the enclosed area.
    
    1. Finds all blue (1) dots in the grid.
    2. Creates a path connecting all blue dots, following the grid edges when possible.
    3. Fills all cells enclosed by the blue path with blue.
    4. Converts red (2) blocks to blue if they are fully enclosed by the blue area.
    5. Preserves original blue dots and leaves unaffected cells unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find blue dots
    blue_dots = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    if len(blue_dots) < 2:
        return output_grid
    
    # Step 2: Create path
    blue_path = create_path(blue_dots, rows, cols)
    
    # Step 3: Fill enclosed area
    fill_enclosed_area(output_grid, blue_path)
    
    # Step 4: Handle red blocks
    handle_red_blocks(input_grid, output_grid)
    
    # Step 5: Preserve original blue dots
    for r, c in blue_dots:
        output_grid.values[r][c] = 1
    
    return output_grid

def create_path(blue_dots: List[Tuple[int, int]], rows: int, cols: int) -> Set[Tuple[int, int]]:
    path = set()
    sorted_dots = sorted(blue_dots)
    
    for i in range(len(sorted_dots)):
        start = sorted_dots[i]
        end = sorted_dots[(i + 1) % len(sorted_dots)]
        path.update(create_edge_path(start, end, rows, cols))
    
    return path

def create_edge_path(start: Tuple[int, int], end: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
    path = []
    r1, c1 = start
    r2, c2 = end
    
    # Move to the edge first
    if abs(c1 - c2) > abs(r1 - r2):
        c1 = 0 if c1 < cols // 2 else cols - 1
    else:
        r1 = 0 if r1 < rows // 2 else rows - 1
    
    path.append((r1, c1))
    
    # Then follow the edge
    while (r1, c1) != (r2, c2):
        if r1 == 0 and c1 != c2:
            c1 += 1 if c2 > c1 else -1
        elif r1 == rows - 1 and c1 != c2:
            c1 += 1 if c2 > c1 else -1
        elif c1 == 0 and r1 != r2:
            r1 += 1 if r2 > r1 else -1
        elif c1 == cols - 1 and r1 != r2:
            r1 += 1 if r2 > r1 else -1
        else:
            if abs(c1 - c2) > abs(r1 - r2):
                c1 += 1 if c2 > c1 else -1
            else:
                r1 += 1 if r2 > r1 else -1
        path.append((r1, c1))
    
    return path

def fill_enclosed_area(grid: ColoredGrid, path: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    queue = deque([(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)])
    outside = set(queue)
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in outside and (nr, nc) not in path:
                outside.add((nr, nc))
                queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in outside and (r, c) not in path:
                grid.values[r][c] = 1

def handle_red_blocks(input_grid: ColoredGrid, output_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2:
                if all(output_grid.values[nr][nc] == 1 
                       for nr in range(max(0, r-1), min(rows, r+2))
                       for nc in range(max(0, c-1), min(cols, c+2))
                       if (nr, nc) != (r, c)):
                    output_grid.values[r][c] = 1
                else:
                    output_grid.values[r][c] = 2
