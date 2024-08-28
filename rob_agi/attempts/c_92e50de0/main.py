from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_92e50de0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by replicating a pattern found in the grid
    based on the following steps:
    1. Analyzes the input grid to determine dimensions, cell size, and dividing line color.
    2. Locates and extracts the pattern from the grid.
    3. Determines the replication structure (frequency and region).
    4. Creates a new grid with the same dimensions and dividing lines as the input.
    5. Replicates the pattern in the appropriate cells, maintaining its original position within each cell.
    6. Returns the new grid as the solution.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the replicated pattern.
    """
    # Step 1: Analyze the input grid
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    dividing_color = max(set(input_grid.values[3]) - {0}, key=lambda x: input_grid.values[3].count(x))
    cell_size = next(i for i in range(1, min(rows, cols)) if input_grid.values[i][0] == dividing_color)

    # Step 2: Locate and extract the pattern
    pattern, pattern_pos = find_pattern(input_grid.values, dividing_color, cell_size)

    # Step 3: Determine replication structure
    replication_freq = determine_replication_frequency(input_grid.values, pattern, pattern_pos, cell_size)

    # Step 4: Create a new grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == dividing_color:
                new_grid[r][c] = dividing_color

    # Step 5: Replicate the pattern
    for cell_row in range(rows // cell_size):
        for cell_col in range(cols // cell_size):
            if (cell_row % replication_freq[0] == pattern_pos[0] % replication_freq[0] and
                cell_col % replication_freq[1] == pattern_pos[1] % replication_freq[1]):
                base_row = cell_row * cell_size
                base_col = cell_col * cell_size
                for r, c, color in pattern:
                    new_row = base_row + r
                    new_col = base_col + c
                    if 0 <= new_row < rows and 0 <= new_col < cols and new_grid[new_row][new_col] != dividing_color:
                        new_grid[new_row][new_col] = color

    # Step 6: Return the new grid
    return ColoredGrid(values=new_grid)

def find_pattern(grid: List[List[int]], dividing_color: int, cell_size: int) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    for r in range(0, rows, cell_size):
        for c in range(0, cols, cell_size):
            pattern = [(i, j, grid[r+i][c+j])
                       for i in range(cell_size) for j in range(cell_size)
                       if grid[r+i][c+j] not in (0, dividing_color)]
            if pattern:
                return pattern, (r // cell_size, c // cell_size)
    raise ValueError("No pattern found in the grid")

def determine_replication_frequency(grid: List[List[int]], pattern: List[Tuple[int, int, int]], 
                                    pattern_pos: Tuple[int, int], cell_size: int) -> Tuple[int, int]:
    rows, cols = len(grid) // cell_size, len(grid[0]) // cell_size
    freq_row = rows
    freq_col = cols
    for r in range(rows):
        for c in range(cols):
            if all(grid[r*cell_size+i][c*cell_size+j] == color 
                   for i, j, color in pattern if color != 0):
                freq_row = min(freq_row, abs(r - pattern_pos[0]))
                freq_col = min(freq_col, abs(c - pattern_pos[1]))
    return max(1, freq_row), max(1, freq_col)
