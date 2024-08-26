from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ea959feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the correct repeating pattern
    and applying it consistently across the entire grid.
    
    The solution works as follows:
    1. Identifies the correct pattern by analyzing the top-left corner of the input grid.
    2. Creates a pattern application function that returns the correct color for any given position.
    3. Generates a new grid by applying the pattern function to each cell.
    4. Returns a new ColoredGrid with the correct pattern applied.
    
    This approach works for all cases by identifying the underlying pattern
    and replicating it across the entire grid, effectively removing any irregularities.
    """
    def identify_pattern(grid: List[List[int]]) -> Tuple[List[List[int]], int, int]:
        rows, cols = len(grid), len(grid[0])
        for pattern_height in range(1, rows + 1):
            for pattern_width in range(1, cols + 1):
                pattern = [row[:pattern_width] for row in grid[:pattern_height]]
                if all(grid[r][c] == pattern[r % pattern_height][c % pattern_width]
                       for r in range(rows) for c in range(cols)
                       if grid[r][c] != 1):  # Ignore blue cells (1) when checking pattern
                    return pattern, pattern_height, pattern_width
        raise ValueError("Could not identify a consistent pattern")

    pattern, pattern_height, pattern_width = identify_pattern(input_grid.values)
    
    def apply_pattern(row: int, col: int) -> int:
        return pattern[row % pattern_height][col % pattern_width]
    
    rows, cols = input_grid.get_dimensions()
    corrected_values = [
        [apply_pattern(i, j) for j in range(cols)]
        for i in range(rows)
    ]
    
    return ColoredGrid(values=corrected_values)
