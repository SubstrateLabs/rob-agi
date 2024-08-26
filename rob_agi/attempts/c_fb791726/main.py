from rob_agi.colored_grid import ColoredGrid

def solve_fb791726(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Doubling its size in both dimensions
    2. Copying non-black cells to their new positions:
       - Top-left quadrant stays in place
       - Top-right quadrant moves to bottom-left
       - Bottom-left quadrant moves to top-right
       - Bottom-right quadrant stays in place
    3. Adding green (3) separator rows and columns between the original rows and columns
    4. Filling the rest with black (0)
    """
    # Step 1: Initialize the output grid
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    # Step 2: Process non-black cells from the input grid
    for i in range(input_rows):
        for j in range(input_cols):
            if input_grid.values[i][j] != 0:
                new_i = 2 * i + (1 if i >= input_rows // 2 else 0)
                new_j = 2 * j + (1 if j >= input_cols // 2 else 0)
                output_grid.values[new_i][new_j] = input_grid.values[i][j]

    # Step 3: Add green separators
    for i in range(1, output_rows, 2):
        output_grid.values[i] = [3] * output_cols
    for j in range(1, output_cols, 2):
        for i in range(output_rows):
            output_grid.values[i][j] = 3

    # Step 4: Return the completed output grid
    return output_grid
