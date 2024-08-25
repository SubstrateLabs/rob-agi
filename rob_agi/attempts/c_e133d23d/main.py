from rob_agi.colored_grid import ColoredGrid

def solve_e133d23d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x3 input grid into a 3x3 output grid based on color patterns.
    
    For each row in the input:
    - Checks if magenta (6) or sky blue (8) is present anywhere in the row for the first output column
    - Checks if yellow (4) is present anywhere in the row for the second output column
    - Checks if magenta (6) or sky blue (8) is present anywhere in the row for the third output column
    - If the condition is met, sets the corresponding output cell to red (2)
    - Otherwise, sets the corresponding output cell to black (0)
    
    Returns the resulting 3x3 ColoredGrid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(3)] for _ in range(3)])
    MAGENTA, YELLOW, SKY_BLUE = 6, 4, 8
    
    for row in range(3):
        input_row = input_grid.values[row]
        
        # Process first column of output
        output_grid.values[row][0] = 2 if any(color in input_row for color in [MAGENTA, SKY_BLUE]) else 0
        
        # Process second column of output
        output_grid.values[row][1] = 2 if YELLOW in input_row else 0
        
        # Process third column of output
        output_grid.values[row][2] = 2 if any(color in input_row for color in [MAGENTA, SKY_BLUE]) else 0
    
    return output_grid
