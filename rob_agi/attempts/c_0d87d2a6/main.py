from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0d87d2a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting blue dots with an efficient path and filling the left area.
    
    1. Finds all blue (1) dots in the grid.
    2. Creates an efficient path connecting all blue dots, preferring edge paths when possible.
    3. Fills all cells to the left of the leftmost blue path with blue.
    4. Converts red (2) blocks to blue if they intersect with the blue path.
    5. Preserves original blue dots and leaves unaffected cells unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find blue dots
    blue_dots = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    if not blue_dots:
        return output_grid
    
    # Step 2: Create efficient path
    blue_dots.sort(key=lambda x: (x[1], x[0]))  # Sort by column, then row
    blue_path = set()
    
    for i in range(len(blue_dots) - 1):
        start, end = blue_dots[i], blue_dots[i+1]
        path = create_path(start, end, rows, cols)
        blue_path.update(path)
    
    # Add blue dots to the path
    blue_path.update(blue_dots)
    
    # Step 3: Fill left area and draw blue path
    leftmost_col = min(c for _, c in blue_path)
    for r in range(rows):
        for c in range(cols):
            if c < leftmost_col or (r, c) in blue_path:
                output_grid.values[r][c] = 1
    
    # Step 4: Handle red blocks
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 2 and (r, c) not in blue_path:
                output_grid.values[r][c] = 2
    
    return output_grid

def create_path(start: Tuple[int, int], end: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
    path = []
    r1, c1 = start
    r2, c2 = end
    
    # Decide whether to move horizontally or vertically first
    if abs(c1 - c2) > abs(r1 - r2):
        # Move horizontally first
        while c1 != c2:
            c1 += 1 if c2 > c1 else -1
            path.append((r1, c1))
        while r1 != r2:
            r1 += 1 if r2 > r1 else -1
            path.append((r1, c1))
    else:
        # Move vertically first
        while r1 != r2:
            r1 += 1 if r2 > r1 else -1
            path.append((r1, c1))
        while c1 != c2:
            c1 += 1 if c2 > c1 else -1
            path.append((r1, c1))
    
    return path
