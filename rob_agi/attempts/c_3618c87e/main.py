from rob_agi.colored_grid import ColoredGrid

def solve_3618c87e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. The top three rows become all zeros.
    2. The fourth row (index 3) only keeps 5s from the input.
    3. The bottom row starts as a copy of the input's bottom row.
    4. 1s from the third row of the input replace 5s in their respective columns in the bottom row of the output.
    """
    # Create a new 5x5 grid filled with zeros
    output = [[0 for _ in range(5)] for _ in range(5)]
    
    # Copy the bottom row from input to output
    output[4] = input_grid.values[4].copy()
    
    # Process the third row (index 2) of the input
    for col in range(5):
        if input_grid.values[2][col] == 1:
            output[4][col] = 1
    
    # Process the fourth row (index 3) of the input
    for col in range(5):
        if input_grid.values[3][col] == 5:
            output[3][col] = 5
    
    return ColoredGrid(values=output)
