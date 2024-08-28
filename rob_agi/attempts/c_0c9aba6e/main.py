from rob_agi.colored_grid import ColoredGrid

def solve_0c9aba6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 13x4 input grid into a 6x4 output grid based on the following rules:
    1. Only the first 6 rows of the input grid are considered.
    2. For each cell in the output grid:
       - Check a 2x2 region in the input grid (current cell, right, below, and diagonal).
       - If there's exactly one red (2) cell in this 2x2 region AND it's in the bottom-right position,
         set the output cell to sky blue (8).
       - Otherwise, set the output cell to black (0).
    3. Returns the resulting 6x4 grid.
    """
    # Create a new 6x4 ColoredGrid for the output, initially filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(6)])

    def check_2x2_region(top_left, top_right, bottom_left, bottom_right):
        red_count = sum(1 for cell in [top_left, top_right, bottom_left, bottom_right] if cell == 2)
        return red_count == 1 and bottom_right == 2

    # Iterate through each cell in the output grid
    for r in range(6):
        for c in range(4):
            top_left = input_grid.values[r][c]
            top_right = input_grid.values[r][c+1] if c < 3 else 0
            bottom_left = input_grid.values[r+1][c] if r < 5 else 0
            bottom_right = input_grid.values[r+1][c+1] if r < 5 and c < 3 else 0

            # Apply the transformation rule
            if check_2x2_region(top_left, top_right, bottom_left, bottom_right):
                output_grid.values[r][c] = 8  # sky blue
            # Otherwise, it remains 0 (black)

    return output_grid
