from rob_agi.colored_grid import ColoredGrid
import copy
from collections import deque

RED, YELLOW, GRAY, BLUE = 2, 4, 5, 8
RED_DIRECTION = (1, -1)  # Down-left
YELLOW_DIRECTION = (-1, 1)  # Up-right

def is_valid_cell(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])

def is_empty_cell(grid, row, col, color):
    return grid[row][col] == 0 or (color == YELLOW and grid[row][col] == RED)

def propagate_color(grid, start_row, start_col, color, direction):
    queue = deque([(start_row, start_col, 0)])
    marked_cells = set()
    
    while queue:
        row, col, distance = queue.popleft()
        if distance > 4:
            continue
        
        if is_valid_cell(grid, row, col) and is_empty_cell(grid, row, col, color):
            marked_cells.add((row, col))
            if distance < 4:
                next_row, next_col = row + direction[0], col + direction[1]
                queue.append((next_row, next_col, distance + 1))
    
    return marked_cells

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating colors from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Propagate red (2) color diagonally down-left up to 4 steps.
    2. Propagate yellow (4) color diagonally up-right up to 4 steps.
    3. Apply red first, then yellow (yellow overwrites red).
    4. Stop propagation at grid edges, non-empty cells, or after 4 steps.
    5. Preserve original gray (5) and blue (8) squares.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid.values)
    rows, cols = len(grid), len(grid[0])
    
    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == GRAY]
    gray_squares.sort(key=lambda x: (-x[0], -x[1]))  # Sort bottom-right to top-left
    
    for row, col in gray_squares:
        red_cells = propagate_color(grid, row, col, RED, RED_DIRECTION)
        yellow_cells = propagate_color(grid, row, col, YELLOW, YELLOW_DIRECTION)
        
        for r, c in red_cells:
            if grid[r][c] == 0:
                grid[r][c] = RED
        
        for r, c in yellow_cells:
            if grid[r][c] == 0 or grid[r][c] == RED:
                grid[r][c] = YELLOW
    
    # Restore original gray and blue squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] in [GRAY, BLUE]:
                grid[r][c] = input_grid.values[r][c]
    
    return ColoredGrid(values=grid)
