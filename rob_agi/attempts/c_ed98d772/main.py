from rob_agi.colored_grid import ColoredGrid

def solve_ed98d772(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by:
    1. Copying the input to the top-left quadrant.
    2. Creating a frame around the grid using non-zero colors from the input.
    3. Filling the interior based on the input's zero and non-zero values and frame colors.
    4. Applying symmetry to complete the grid.
    """
    # Create the initial 6x6 output grid
    output_grid = [[0 for _ in range(6)] for _ in range(6)]

    # Step 1: Copy input to top-left quadrant
    for r in range(3):
        for c in range(3):
            output_grid[r][c] = input_grid.values[r][c]

    # Step 2: Create the frame
    for i in range(6):
        output_grid[0][i] = output_grid[0][i] if i < 3 else output_grid[0][5-i]
        output_grid[5][i] = output_grid[0][i]
        output_grid[i][0] = output_grid[i][0] if i < 3 else output_grid[5-i][0]
        output_grid[i][5] = output_grid[i][0]

    # Step 3: Fill the interior
    for r in range(1, 5):
        for c in range(1, 5):
            if output_grid[r % 3][c % 3] != 0:
                # Non-zero values from input create "holes"
                output_grid[r][c] = 0
            else:
                # Fill with the corresponding frame color
                output_grid[r][c] = output_grid[0][c] if output_grid[0][c] != 0 else output_grid[r][0]

    # Step 4: Apply symmetry
    for r in range(3):
        for c in range(3):
            output_grid[r][5-c] = output_grid[r][c]
            output_grid[5-r][c] = output_grid[r][c]
            output_grid[5-r][5-c] = output_grid[r][c]

    return ColoredGrid(values=output_grid)
