from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Callable

def detect_pattern(grid: ColoredGrid) -> List[int]:
    """Detect the pattern sequence in the input grid."""
    pattern = []
    for row in range(2, grid.num_rows):
        for col in range(2, grid.num_cols):
            if grid[row][col] not in [0, 1, 4]:
                pattern.append(grid[row][col])
                if len(pattern) > 1 and pattern == pattern[:len(pattern)//2]*2:
                    return pattern[:len(pattern)//2]
    return pattern

def create_pattern_generator(sequence: List[int], grid_width: int) -> Callable[[int, int], int]:
    """Create a generator function for the pattern sequence."""
    def generator(row: int, col: int) -> int:
        pattern_index = ((row - 2) * (grid_width - 2) + (col - 2)) % len(sequence)
        return sequence[pattern_index]
    return generator

def solve_4aab4007(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by filling black areas with a detected pattern.
    
    The function detects the repeating color pattern in the non-black areas of the input grid,
    creates a new grid with the same blue and yellow borders, and fills the
    inner area with the detected pattern, effectively removing all black regions.
    The pattern continues across rows as if it filled the entire inner area.
    """
    pattern = detect_pattern(input_grid)
    output_grid = ColoredGrid(values=[row[:] for row in input_grid.values])
    
    pattern_generator = create_pattern_generator(pattern, input_grid.num_cols)
    
    for row in range(2, output_grid.num_rows):
        for col in range(2, output_grid.num_cols):
            if input_grid[row][col] in [0, 4]:
                output_grid.values[row][col] = pattern_generator(row, col)
            else:
                output_grid.values[row][col] = input_grid[row][col]
    
    return output_grid
