from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_0d87d2a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue dots with a path and filling areas.
    
    1. Finds all blue (1) dots in the grid.
    2. Creates a path connecting all blue dots, prioritizing left and top movements.
    3. Fills all cells to the left of the blue path and inside enclosed areas with blue.
    4. Preserves original blue dots and red blocks.
    5. Handles edge cases like single blue dot, dots on edges, or no blue dots.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find blue dots
    blue_dots = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    if not blue_dots:
        return output_grid
    
    # Step 2: Create path connecting blue dots
    blue_dots.sort(key=lambda x: (x[1], x[0]))  # Sort by column first, then row
    for i in range(len(blue_dots) - 1):
        r1, c1 = blue_dots[i]
        r2, c2 = blue_dots[i + 1]
        # Move left
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if input_grid.values[r1][c] != 2:  # Avoid overwriting red blocks
                output_grid.values[r1][c] = 1
        # Move vertically
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if input_grid.values[r][c2] != 2:  # Avoid overwriting red blocks
                output_grid.values[r][c2] = 1
    
    # Step 3: Fill areas
    for r in range(rows):
        left_blue = -1
        for c in range(cols):
            if output_grid.values[r][c] == 1:
                left_blue = c
            elif left_blue != -1 and input_grid.values[r][c] != 2:
                output_grid.values[r][c] = 1
    
    # Step 4: Fill enclosed areas
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 0 and (r, c) not in visited:
                area = []
                queue = deque([(r, c)])
                enclosed = True
                while queue:
                    cr, cc = queue.popleft()
                    if (cr, cc) in visited:
                        continue
                    visited.add((cr, cc))
                    if output_grid.values[cr][cc] == 0:
                        area.append((cr, cc))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                queue.append((nr, nc))
                            else:
                                enclosed = False
                    elif output_grid.values[cr][cc] == 2:
                        enclosed = False
                if enclosed:
                    for ar, ac in area:
                        output_grid.values[ar][ac] = 1
    
    # Step 5: Preserve original elements
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                output_grid.values[r][c] = 1
            elif input_grid.values[r][c] == 2:
                output_grid.values[r][c] = 2
    
    return output_grid
