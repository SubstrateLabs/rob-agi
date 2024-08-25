from rob_agi.colored_grid import ColoredGrid

def solve_e133d23d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x3 input grid into a 3x3 output grid based on color patterns.
    
    For each row in the input:
    - Checks if magenta (6) or sky blue (8) is present in the first three columns for the first output column
    - Checks if magenta (6) or sky blue (8) is present in the middle three columns for the second output column
    - Checks if magenta (6) or sky blue (8) is present in the last three columns for the third output column
    - If the condition is met, sets the corresponding output cell to red (2)
    - Otherwise, sets the corresponding output cell to black (0)
    
    Returns the resulting 3x3 ColoredGrid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(3)] for _ in range(3)])
    
    def check_magenta_or_sky(sublist):
        return 6 in sublist or 8 in sublist
    
    for row in range(3):
        input_row = input_grid.values[row]
        
        # Process first column of output
        output_grid.values[row][0] = 2 if check_magenta_or_sky(input_row[:3]) else 0
        
        # Process second column of output
        output_grid.values[row][1] = 2 if check_magenta_or_sky(input_row[2:5]) else 0
        
        # Process third column of output
        output_grid.values[row][2] = 2 if check_magenta_or_sky(input_row[4:]) else 0
    
    return output_grid
