from rob_agi.colored_grid import ColoredGrid

def solve_f0afb749(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Doubling the grid size
    2. Expanding non-black squares into 2x2 blocks
    3. Adding blue squares (1s) in empty spaces:
       - Top-left to bottom-right diagonal in the top-left quadrant
       - Bottom-right to top-left diagonal in the bottom-right quadrant
    """
    # Step 1: Initialize the output grid
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    # Step 2: Expand non-black squares
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                output_grid[r*2][c*2] = color
                output_grid[r*2][c*2+1] = color
                output_grid[r*2+1][c*2] = color
                output_grid[r*2+1][c*2+1] = color

    # Step 3: Identify empty rows and columns
    empty_rows = [all(cell == 0 for cell in row) for row in output_grid]
    empty_cols = [all(row[c] == 0 for row in output_grid) for c in range(output_cols)]

    # Step 4: Place blue squares in the top-left quadrant
    r, c = 0, 0
    while r < output_rows // 2 and c < output_cols // 2:
        if empty_rows[r] and empty_cols[c]:
            output_grid[r][c] = 1
        r += 1
        c += 1

    # Step 5: Place blue squares in the bottom-right quadrant
    r, c = output_rows - 1, output_cols - 1
    while r >= output_rows // 2 and c >= output_cols // 2:
        if empty_rows[r] and empty_cols[c]:
            output_grid[r][c] = 1
        r -= 1
        c -= 1

    # Step 6: Create and return the final ColoredGrid
    return ColoredGrid(values=output_grid)
