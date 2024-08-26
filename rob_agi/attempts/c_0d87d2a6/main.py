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
    6. Handles edge cases like single blue dot or two blue dots.
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
    handle_red_blocks(input_grid, output_grid, blue_path)
    
    # Step 5: Preserve original blue dots and path
    for r, c in blue_dots:
        output_grid.values[r][c] = 1
    for r, c in blue_path:
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
    
    # Move to the nearest edge first
    if abs(c1 - c2) > abs(r1 - r2):
        c1 = 0 if c1 < cols // 2 else cols - 1
    else:
        r1 = 0 if r1 < rows // 2 else rows - 1
    
    path.append((r1, c1))
    
    # Then follow the edge
    while (r1, c1) != (r2, c2):
        if r1 == 0 or r1 == rows - 1:
            if c1 != c2:
                c1 += 1 if c2 > c1 else -1
            else:
                r1 += 1 if r2 > r1 else -1
        elif c1 == 0 or c1 == cols - 1:
            if r1 != r2:
                r1 += 1 if r2 > r1 else -1
            else:
                c1 += 1 if c2 > c1 else -1
        else:
            # If we're not on an edge, move towards the target
            if abs(c1 - c2) > abs(r1 - r2):
                c1 += 1 if c2 > c1 else -1
            else:
                r1 += 1 if r2 > r1 else -1
        path.append((r1, c1))
    
    return path

def fill_enclosed_area(grid: ColoredGrid, path: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    outside = set()
    
    def flood_fill(r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in outside and (r, c) not in path:
                outside.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
    
    # Start flood fill from all edges
    for r in [0, rows-1]:
        for c in range(cols):
            if (r, c) not in path:
                flood_fill(r, c)
    for c in [0, cols-1]:
        for r in range(rows):
            if (r, c) not in path:
                flood_fill(r, c)
    
    # Fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in outside and (r, c) not in path:
                grid.values[r][c] = 1

def handle_red_blocks(input_grid: ColoredGrid, output_grid: ColoredGrid, blue_path: Set[Tuple[int, int]]):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2:
                if not any((nr, nc) in blue_path
                           for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                           if 0 <= nr < rows and 0 <= nc < cols):
                    if all(output_grid.values[nr][nc] == 1 
                           for nr in range(max(0, r-1), min(rows, r+2))
                           for nc in range(max(0, c-1), min(cols, c+2))
                           if (nr, nc) != (r, c)):
                        output_grid.values[r][c] = 1
                    else:
                        output_grid.values[r][c] = 2
                else:
                    output_grid.values[r][c] = 2
