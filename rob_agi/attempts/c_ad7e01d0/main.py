from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive approach:
    1. The output grid size is the square of the input grid size.
    2. For odd-sized inputs, fill the middle row and column with the input pattern.
    3. For even-sized inputs, fill the entire perimeter with the input pattern.
    4. Recursively fill the remaining spaces:
       - For odd-sized inputs, fill the bottom-right quadrant.
       - For even-sized inputs, fill the inner area of the frame.
    5. Continue this process until the base case (size <= input size) is reached.
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    def copy_pattern(top: int, left: int, size: int):
        for i in range(size):
            for j in range(size):
                output_values[top + i][left + j] = input_grid.values[i % n][j % n]
    
    def fill_initial_pattern():
        if n % 2 == 1:  # Odd-sized input
            # Fill middle row and column
            mid = output_size // 2
            for i in range(output_size):
                copy_pattern(mid, i, 1)
                copy_pattern(i, mid, 1)
        else:  # Even-sized input
            # Fill entire perimeter
            for i in range(output_size):
                copy_pattern(0, i, n)  # Top
                copy_pattern(output_size - n, i, n)  # Bottom
                copy_pattern(i, 0, n)  # Left
                copy_pattern(i, output_size - n, n)  # Right
    
    def apply_pattern(top: int, left: int, size: int):
        if size <= n:
            copy_pattern(top, left, size)
            return
        
        if n % 2 == 1:  # Odd-sized input
            quadrant_size = size // 2
            apply_pattern(top + quadrant_size + 1, left + quadrant_size + 1, quadrant_size)
        else:  # Even-sized input
            inner_size = size - 2 * n
            if inner_size > 0:
                apply_pattern(top + n, left + n, inner_size)
    
    fill_initial_pattern()
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
