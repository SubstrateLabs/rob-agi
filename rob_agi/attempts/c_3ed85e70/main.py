from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

color_sequence = {1: 2, 2: 4, 4: 8, 8: 1}

def find_2x2_squares(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    squares = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.values[r+dr][c+dc] == color for dr in range(2) for dc in range(2)):
                squares.append((r, c))
    return squares

def expand_2x2_to_4x4(grid: ColoredGrid, row: int, col: int, color: int, border_color: int) -> None:
    for r in range(row-1, row+3):
        for c in range(col-1, col+3):
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                if r == row-1 or r == row+2 or c == col-1 or c == col+2:
                    grid.values[r][c] = border_color
                else:
                    grid.values[r][c] = color

def process_3x3_pattern(grid: ColoredGrid, row: int, col: int) -> None:
    center_color = grid.values[row][col]
    if center_color in color_sequence:
        next_color = color_sequence[center_color]
        for r in range(row-1, row+2):
            for c in range(col-1, col+2):
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    if r == row and c == col:
                        grid.values[r][c] = next_color
                    elif grid.values[r][c] == 0:
                        grid.values[r][c] = center_color

def fill_empty_spaces(grid: ColoredGrid) -> None:
    for r in range(1, grid.num_rows - 1):
        for c in range(1, grid.num_cols - 1):
            if all(grid.values[r+dr][c+dc] == 0 for dr in [-1, 0, 1] for dc in [-1, 0, 1]):
                process_3x3_pattern(grid, r, c)

def solve_3ed85e70(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Expand 2x2 sky (8) squares to 4x4 with blue (1) borders.
    2. Process 3x3 patterns: create new ones or update existing ones following the color sequence.
    3. Expand 2x2 yellow (4) squares to 4x4.
    4. Fill empty 3x3 spaces with new patterns.
    5. Repeat until no more changes can be made.
    
    Color sequence: 1 (blue) -> 2 (red) -> 4 (yellow) -> 8 (sky) -> 1 (blue)
    """
    grid = input_grid.deep_copy()
    unchangeable = set((r, c) for r, row in enumerate(grid.values) 
                       for c, val in enumerate(row) if val == 3)
    
    while True:
        original = grid.deep_copy()
        
        # Expand 2x2 sky squares
        sky_squares = find_2x2_squares(grid, 8)
        for row, col in sky_squares:
            expand_2x2_to_4x4(grid, row, col, 8, 1)
        
        # Process 3x3 patterns
        for r in range(1, grid.num_rows - 1):
            for c in range(1, grid.num_cols - 1):
                if (r, c) not in unchangeable:
                    process_3x3_pattern(grid, r, c)
        
        # Expand 2x2 yellow squares
        yellow_squares = find_2x2_squares(grid, 4)
        for row, col in yellow_squares:
            expand_2x2_to_4x4(grid, row, col, 4, 4)
        
        # Fill empty spaces
        fill_empty_spaces(grid)
        
        if grid.values == original.values:
            break
    
    return grid
