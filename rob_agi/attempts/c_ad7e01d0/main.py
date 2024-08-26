from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive, fractal-like approach:
    1. The output grid size is the square of the input grid size.
    2. For odd-sized inputs, apply the pattern to the left column, top row, and bottom row.
    3. For even-sized inputs, apply the pattern to all four corners and recurse on the center.
    4. Fill the remaining areas based on the pattern observed in the examples.
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    def copy_pattern(top: int, left: int):
        for i in range(n):
            for j in range(n):
                output_values[top + i][left + j] = input_grid.values[i][j]
    
    def apply_pattern(top: int, left: int, size: int):
        if size == n:
            copy_pattern(top, left)
        elif n % 2 == 1:  # Odd-sized input
            # Apply pattern to left column, top row, and bottom row
            for i in range(size):
                output_values[top + i][left] = input_grid.values[i % n][0]
                output_values[top][left + i] = input_grid.values[0][i % n]
                output_values[top + size - 1][left + i] = input_grid.values[n - 1][i % n]
            # Recurse on the remaining part
            if size > n:
                apply_pattern(top + 1, left + 1, size - 2)
        else:  # Even-sized input
            # Apply pattern to corners
            copy_pattern(top, left)
            copy_pattern(top, left + size - n)
            copy_pattern(top + size - n, left)
            copy_pattern(top + size - n, left + size - n)
            if size > 2 * n:
                # Recurse on center
                apply_pattern(top + n, left + n, size - 2 * n)
    
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
