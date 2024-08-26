from rob_agi.colored_grid import ColoredGrid

def solve_40f6cd08(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by creating a symmetrical pattern based on the input grid.
    
    1. Determine the pattern size from the top-left corner
    2. Extract the reference pattern
    3. Create a new 30x30 output grid
    4. Mirror the pattern to all four quadrants
    5. Ensure the central cross (rows and columns 14 and 15) remains black (0)
    6. Return the resulting symmetrical output grid
    
    Returns a new 30x30 ColoredGrid with the transformed symmetrical pattern.
    """
    # Determine pattern size
    pattern_size = 0
    for i in range(14):
        if input_grid.values[i][i] == 0:
            break
        pattern_size = i + 1

    # Extract reference pattern
    reference = input_grid.extract_subgrid(0, 0, pattern_size, pattern_size)

    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])

    # Mirror pattern to all quadrants
    for i in range(pattern_size):
        for j in range(pattern_size):
            value = reference.values[i][j]
            output.values[i][j] = value
            output.values[i][29-j] = value
            output.values[29-i][j] = value
            output.values[29-i][29-j] = value

    # Ensure central cross remains black
    for i in range(30):
        output.values[14][i] = 0
        output.values[15][i] = 0
        output.values[i][14] = 0
        output.values[i][15] = 0

    return output
