from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid based on the following rules:
    1. The output grid size is the square of the input grid size.
    2. Copy the input pattern to all four corners of the output grid.
    3. For odd-sized inputs, fill the middle column of blocks with the input pattern.
    4. For even-sized inputs, fill the top, bottom, left, and right edge blocks with the input pattern.
    5. Leave the remaining areas as zeros (black).
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    # Helper function to copy input pattern to a specific block
    def copy_pattern(start_row, start_col):
        for i in range(n):
            for j in range(n):
                output_values[start_row + i][start_col + j] = input_grid.values[i][j]
    
    # Fill corner blocks
    copy_pattern(0, 0)  # Top-left
    copy_pattern(0, output_size - n)  # Top-right
    copy_pattern(output_size - n, 0)  # Bottom-left
    copy_pattern(output_size - n, output_size - n)  # Bottom-right
    
    if n % 2 == 1:  # Odd-sized input
        # Fill middle column of blocks
        for i in range(1, n - 1):
            copy_pattern(i * n, n)
    else:  # Even-sized input
        # Fill top and bottom edge blocks
        for j in range(1, n - 1):
            copy_pattern(0, j * n)  # Top edge
            copy_pattern(output_size - n, j * n)  # Bottom edge
        
        # Fill left and right edge blocks
        for i in range(1, n - 1):
            copy_pattern(i * n, 0)  # Left edge
            copy_pattern(i * n, output_size - n)  # Right edge
    
    return ColoredGrid(values=output_values)
