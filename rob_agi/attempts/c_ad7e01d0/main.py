from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive, fractal-like approach:
    1. The output grid size is the square of the input grid size.
    2. For odd-sized inputs, apply the pattern only to the center column and row.
    3. For even-sized inputs, apply the pattern to the corners and recurse on the center.
    4. Leave the remaining areas as zeros (black).
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
            mid = size // 2 - n // 2
            # Apply pattern only to center column and row
            apply_pattern(top + mid, left + mid, n)
        else:  # Even-sized input
            # Apply pattern to corners
            apply_pattern(top, left, n)
            apply_pattern(top, left + size - n, n)
            apply_pattern(top + size - n, left, n)
            apply_pattern(top + size - n, left + size - n, n)
            if size > 2 * n:
                # Recurse on center
                apply_pattern(top + n, left + n, size - 2 * n)
    
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
