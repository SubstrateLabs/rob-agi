from rob_agi.colored_grid import ColoredGrid

def solve_40f6cd08(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by replicating the top-left quadrant pattern to other non-empty quadrants.
    
    1. Copy the top-left quadrant (0,0 to 13,13) from input to output grid
    2. Identify non-empty quadrants in the input grid
    3. For each non-empty quadrant, replicate the top-left pattern
    4. Ensure the central cross (rows and columns 14 and 15) remains black (0)
    5. Return the resulting transformed grid
    
    Returns a new 30x30 ColoredGrid with the transformed pattern.
    """
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])

    # Copy top-left quadrant
    for i in range(14):
        for j in range(14):
            output.values[i][j] = input_grid.values[i][j]

    # Define quadrants
    quadrants = [
        ((0, 16), (13, 29)),  # Top-right
        ((16, 0), (29, 13)),  # Bottom-left
        ((16, 16), (29, 29))  # Bottom-right
    ]

    # Process other quadrants
    for (top, left), (bottom, right) in quadrants:
        if any(input_grid.values[i][j] != 0 for i in range(top, bottom+1) for j in range(left, right+1)):
            for i in range(14):
                for j in range(14):
                    output.values[top+i][left+j] = input_grid.values[i][j]

    # Ensure central cross remains black
    for i in range(30):
        output.values[14][i] = 0
        output.values[15][i] = 0
        output.values[i][14] = 0
        output.values[i][15] = 0

    return output
