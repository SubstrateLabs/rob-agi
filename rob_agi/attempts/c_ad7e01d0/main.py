from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive, fractal-like approach:
    1. The output grid size is the square of the input grid size.
    2. Initially repeat the input pattern across the entire output grid.
    3. For odd-sized inputs, keep the left column, top row, and bottom row, then recurse on the inner area.
    4. For even-sized inputs, keep the four corners, then recurse on the center if large enough.
    5. Recursively apply this process until the base case is reached.
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    def copy_pattern(top: int, left: int, size: int):
        for i in range(size):
            for j in range(size):
                output_values[top + i][left + j] = input_grid.values[i % n][j % n]
    
    def apply_pattern(top: int, left: int, size: int):
        if size <= n:
            copy_pattern(top, left, size)
            return
        
        # Initially repeat the pattern across the entire area
        copy_pattern(top, left, size)
        
        if n % 2 == 1:  # Odd-sized input
            # Clear the inner area
            for i in range(top + 1, top + size - 1):
                for j in range(left + 1, left + size - 1):
                    output_values[i][j] = 0
            # Recurse on the inner area
            if size > n + 2:
                apply_pattern(top + 1, left + 1, size - 2)
        else:  # Even-sized input
            # Clear the center
            center_start = size // 2 - n // 2
            center_end = center_start + n
            for i in range(top + center_start, top + center_end):
                for j in range(left + center_start, left + center_end):
                    output_values[i][j] = 0
            # Recurse on the center if large enough
            if size > 2 * n:
                apply_pattern(top + center_start, left + center_start, n)
    
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
