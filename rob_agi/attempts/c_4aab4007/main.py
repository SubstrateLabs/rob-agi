from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Callable

def detect_pattern(grid: ColoredGrid) -> List[int]:
    """Detect the pattern sequence in the input grid."""
    pattern = []
    for row in range(2, grid.num_rows):
        for col in range(2, grid.num_cols):
            if grid[row][col] not in [0, 1, 4] and grid[row][col] not in pattern:
                pattern.append(grid[row][col])
    return pattern

def create_pattern_generator(sequence: List[int]) -> Callable[[], int]:
    """Create a generator function for the pattern sequence."""
    def generator():
        i = 0
        while True:
            yield sequence[i]
            i = (i + 1) % len(sequence)
    return generator().__next__

def fill_inner_area(grid: ColoredGrid, pattern_generator: Callable[[], int], start_row: int, start_col: int, end_row: int, end_col: int) -> None:
    """Fill the inner area of the grid with the pattern."""
    for i in range(end_row - start_row):
        for j in range(end_col - start_col):
            row = start_row + i
            col = start_col + (i + j) % (end_col - start_col)
            grid.values[row][col] = pattern_generator()

def solve_4aab4007(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by filling black areas with a detected pattern.
    
    The function detects the pattern in the non-black areas of the input grid,
    creates a new grid with the same blue and yellow borders, and fills the
    inner area with the detected pattern, effectively removing all black regions.
    """
    pattern = detect_pattern(input_grid)
    output_grid = ColoredGrid(values=[row[:] for row in input_grid.values])
    
    # Copy blue border
    for i in range(2):
        for j in range(output_grid.num_cols):
            output_grid.values[i][j] = 1
    for i in range(output_grid.num_rows):
        for j in range(2):
            output_grid.values[i][j] = 1
    
    # Copy yellow border
    for i in range(2, output_grid.num_rows):
        output_grid.values[i][2] = 4
    for j in range(2, output_grid.num_cols):
        output_grid.values[2][j] = 4
    
    pattern_generator = create_pattern_generator(pattern)
    fill_inner_area(output_grid, pattern_generator, 3, 3, output_grid.num_rows, output_grid.num_cols)
    
    return output_grid
