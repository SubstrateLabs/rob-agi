from rob_agi.colored_grid import ColoredGrid

def solve_4852f2fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the number of yellow squares.
    
    1. Count yellow (4) squares in the input grid.
    2. Create a 3xN output grid where N = (yellow_count + 1) * 3.
    3. Fill the grid with a pattern of 3x3 "framed" squares.
    4. Invert the middle row of the pattern.
    5. Trim the output if necessary to match input width.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Count yellow squares
    yellow_count = sum(row.count(4) for row in input_grid.values)
    
    # Determine output size
    output_width = (yellow_count + 1) * 3
    
    # Create base output grid
    output = [
        [8, 8, 0] * (yellow_count + 1),
        [8, 0, 8] * (yellow_count + 1),
        [8, 8, 0] * (yellow_count + 1)
    ]
    
    # Invert middle row
    output[1] = [8 if x == 0 else 0 for x in output[1]]
    
    # Trim if necessary
    input_width = len(input_grid.values[0])
    while len(output[0]) > input_width:
        for row in output:
            row.pop()
            row.pop()
            row.pop()
    
    return ColoredGrid(values=output)
