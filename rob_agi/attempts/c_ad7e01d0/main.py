from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid using a recursive, fractal-like approach:
    1. The output grid size is the square of the input grid size.
    2. Apply the input pattern recursively to create a fractal-like structure.
    3. For odd-sized inputs, fill the center and middle edges with the pattern.
    4. For even-sized inputs, fill the edges with the pattern and recurse on the center.
    5. Leave the remaining areas as zeros (black).
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
        else:
            # Apply pattern to corners
            apply_pattern(top, left, n)
            apply_pattern(top, left + size - n, n)
            apply_pattern(top + size - n, left, n)
            apply_pattern(top + size - n, left + size - n, n)
            
            if n % 2 == 1:  # Odd-sized input
                # Apply pattern to middle of edges and center
                mid = (size - n) // 2
                apply_pattern(top + mid, left, n)
                apply_pattern(top + mid, left + size - n, n)
                apply_pattern(top, left + mid, n)
                apply_pattern(top + size - n, left + mid, n)
                apply_pattern(top + mid, left + mid, n)
            elif size > 2 * n:  # Even-sized input, continue recursion
                # Apply pattern to all edge blocks
                for i in range(1, size // n - 1):
                    apply_pattern(top, left + i * n, n)
                    apply_pattern(top + size - n, left + i * n, n)
                    apply_pattern(top + i * n, left, n)
                    apply_pattern(top + i * n, left + size - n, n)
                # Recurse on center
                apply_pattern(top + n, left + n, size - 2 * n)
    
    apply_pattern(0, 0, output_size)
    return ColoredGrid(values=output_values)
