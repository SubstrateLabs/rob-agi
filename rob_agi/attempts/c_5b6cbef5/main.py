from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by expanding the pattern.
    
    The solution involves the following steps:
    1. Analyze the input 4x4 grid to determine the quadrant with the most non-zero elements.
    2. Create a 16x16 output grid, placing a 2x2 arrangement of the original pattern
       in either the top-left or bottom-right quadrant based on the analysis.
    3. Fill the remaining quadrants with appropriate copies or variations of the original pattern.
    4. Refine the pattern by preserving any all-zero rows or columns from the original input.
    
    This process creates a larger pattern that preserves the input's structure
    while expanding it across the 16x16 grid, maintaining the original pattern's characteristics.
    """
    def count_non_zero(grid, start_row, start_col, size):
        return sum(grid[r][c] != 0 
                   for r in range(start_row, start_row + size)
                   for c in range(start_col, start_col + size))

    def copy_pattern(src, dest, src_row, src_col, dest_row, dest_col, size):
        for i in range(size):
            for j in range(size):
                dest[dest_row + i][dest_col + j] = src[src_row + i][src_col + j]

    # Analyze the input grid
    bottom_right_count = count_non_zero(input_grid.values, 2, 2, 2)
    total_count = count_non_zero(input_grid.values, 0, 0, 4)
    place_bottom_right = bottom_right_count > total_count // 2

    # Create the 16x16 output grid
    output = [[0 for _ in range(16)] for _ in range(16)]

    # Place the 2x2 arrangement
    if place_bottom_right:
        for i in range(2):
            for j in range(2):
                copy_pattern(input_grid.values, output, 0, 0, 8 + i*4, 8 + j*4, 4)
    else:
        for i in range(2):
            for j in range(2):
                copy_pattern(input_grid.values, output, 0, 0, i*4, j*4, 4)

    # Fill the remaining quadrants
    if place_bottom_right:
        copy_pattern(input_grid.values, output, 0, 0, 0, 8, 4)  # Top-right
        copy_pattern(input_grid.values, output, 0, 0, 8, 0, 4)  # Bottom-left
    else:
        copy_pattern(input_grid.values, output, 0, 0, 0, 12, 4)  # Top-right
        copy_pattern(input_grid.values, output, 0, 0, 12, 0, 4)  # Bottom-left
        # Bottom-right: copy non-zero elements
        for i in range(4):
            for j in range(4):
                if input_grid.values[i][j] != 0:
                    output[12 + i][12 + j] = input_grid.values[i][j]

    # Refine the pattern
    for i in range(4):
        if all(input_grid.values[i][j] == 0 for j in range(4)):
            for j in range(16):
                output[i][j] = output[i + 4][j] = output[i + 8][j] = output[i + 12][j] = 0
        if all(input_grid.values[j][i] == 0 for j in range(4)):
            for j in range(16):
                output[j][i] = output[j][i + 4] = output[j][i + 8] = output[j][i + 12] = 0

    return ColoredGrid(values=output)
