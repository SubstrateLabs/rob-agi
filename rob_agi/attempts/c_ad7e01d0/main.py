from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive, fractal-like approach:
    1. The output grid size is the square of the input grid size.
    2. Initially repeat the input pattern across the entire output grid.
    3. For odd-sized inputs, clear the inner area leaving a frame of width 1, then recurse on the inner area.
    4. For even-sized inputs, divide into four quadrants, clear the inner area of each quadrant leaving a frame of width 1, then recurse on each cleared inner area.
    5. Recursively apply this process until the base case (size <= input size) is reached.
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
            quadrant_size = size // 2
            for quad_top in [top, top + quadrant_size]:
                for quad_left in [left, left + quadrant_size]:
                    # Clear the inner area of each quadrant
                    for i in range(quad_top + 1, quad_top + quadrant_size - 1):
                        for j in range(quad_left + 1, quad_left + quadrant_size - 1):
                            output_values[i][j] = 0
                    # Recurse on the cleared inner area of each quadrant
                    if quadrant_size > n + 2:
                        apply_pattern(quad_top + 1, quad_left + 1, quadrant_size - 2)
    
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
