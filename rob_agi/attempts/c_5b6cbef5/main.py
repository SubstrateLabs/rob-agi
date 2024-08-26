from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by expanding the pattern.
    
    The solution involves the following steps:
    1. Create an 8x8 expanded pattern from the 4x4 input
    2. Use the 8x8 pattern to fill a 16x16 grid:
       - Place the 8x8 pattern in the center
       - Copy appropriate sections to the corners and edges
    
    This process creates a larger pattern that preserves the input's structure
    while expanding it across the 16x16 grid, maintaining symmetry and proper spacing.
    """
    def create_8x8_pattern(input_grid):
        pattern = [[0 for _ in range(8)] for _ in range(8)]
        for i in range(4):
            for j in range(4):
                r, c = i * 2, j * 2
                pattern[r][c] = input_grid.values[i][j]
                if input_grid.values[i][j] != 0:
                    if i < 3 and input_grid.values[i+1][j] != 0:
                        pattern[r+1][c] = input_grid.values[i][j]
                    if j < 3 and input_grid.values[i][j+1] != 0:
                        pattern[r][c+1] = input_grid.values[i][j]
                    if i < 3 and j < 3 and input_grid.values[i+1][j+1] != 0:
                        pattern[r+1][c+1] = input_grid.values[i][j]
        return pattern

    def fill_16x16_grid(pattern):
        grid = [[0 for _ in range(16)] for _ in range(16)]
        # Fill center
        for i in range(8):
            for j in range(8):
                grid[i+4][j+4] = pattern[i][j]
        # Fill corners
        for i in range(4):
            for j in range(4):
                grid[i][j] = pattern[i][j]
                grid[i][j+12] = pattern[i][j+4]
                grid[i+12][j] = pattern[i+4][j]
                grid[i+12][j+12] = pattern[i+4][j+4]
        # Fill edges
        for i in range(8):
            for j in range(4):
                grid[i+4][j] = pattern[i][j]
                grid[i+4][j+12] = pattern[i][j+4]
        for i in range(4):
            for j in range(8):
                grid[i][j+4] = pattern[i][j]
                grid[i+12][j+4] = pattern[i+4][j]
        return grid

    pattern_8x8 = create_8x8_pattern(input_grid)
    output_grid = fill_16x16_grid(pattern_8x8)
    return ColoredGrid(values=output_grid)
