from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Callable

def detect_pattern(grid: ColoredGrid) -> Tuple[List[int], str]:
    """Detect the pattern sequence and type in the input grid."""
    pattern = []
    rows, cols = grid.get_dimensions()
    
    # Check for spiral pattern
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    r, c = 2, 2
    dir_index = 0
    visited = set()
    
    while len(visited) < (rows - 2) * (cols - 2):
        if grid[r][c] not in [0, 1, 4]:
            pattern.append(grid[r][c])
        visited.add((r, c))
        
        next_r, next_c = r + directions[dir_index][0], c + directions[dir_index][1]
        if (next_r < 2 or next_r >= rows or next_c < 2 or next_c >= cols or 
            (next_r, next_c) in visited):
            dir_index = (dir_index + 1) % 4
            next_r, next_c = r + directions[dir_index][0], c + directions[dir_index][1]
        
        r, c = next_r, next_c
    
    if len(set(pattern)) == len(pattern):
        return pattern, "spiral"
    
    # Check for horizontal pattern
    pattern = []
    for r in range(2, rows):
        for c in range(2, cols):
            if grid[r][c] not in [0, 1, 4]:
                pattern.append(grid[r][c])
                if len(pattern) > 1 and pattern == pattern[:len(pattern)//2]*2:
                    return pattern[:len(pattern)//2], "horizontal"
    
    # If no clear pattern is found, return all non-black, non-yellow colors
    pattern = list(set([grid[r][c] for r in range(2, rows) for c in range(2, cols) 
                        if grid[r][c] not in [0, 1, 4]]))
    return pattern, "default"

def create_pattern_generator(sequence: List[int], grid_width: int, pattern_type: str) -> Callable[[int, int], int]:
    """Create a generator function for the pattern sequence."""
    if pattern_type == "spiral":
        def generator(row: int, col: int) -> int:
            index = min(row - 2, col - 2, grid_width - 1 - col, grid_width - 1 - row)
            return sequence[index % len(sequence)]
    elif pattern_type == "horizontal":
        def generator(row: int, col: int) -> int:
            pattern_index = ((row - 2) * (grid_width - 2) + (col - 2)) % len(sequence)
            return sequence[pattern_index]
    else:  # default
        def generator(row: int, col: int) -> int:
            return sequence[(row + col) % len(sequence)]
    return generator

def solve_4aab4007(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by filling black areas with a detected pattern.
    
    The function detects the repeating color pattern and its type (spiral, horizontal, or default)
    in the non-black areas of the input grid. It then creates a new grid with the same blue and
    yellow borders, and fills the inner area with the detected pattern, effectively removing all
    black regions. The pattern continues based on its detected type across the entire inner area.
    """
    pattern, pattern_type = detect_pattern(input_grid)
    output_grid = ColoredGrid(values=[row[:] for row in input_grid.values])
    
    pattern_generator = create_pattern_generator(pattern, input_grid.num_cols, pattern_type)
    
    for row in range(2, output_grid.num_rows):
        for col in range(2, output_grid.num_cols):
            if input_grid[row][col] in [0, 4]:
                output_grid.values[row][col] = pattern_generator(row, col)
            else:
                output_grid.values[row][col] = input_grid[row][col]
    
    return output_grid
