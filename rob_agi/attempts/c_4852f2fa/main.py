from rob_agi.colored_grid import ColoredGrid

def solve_4852f2fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the number of yellow squares.
    
    1. Count yellow (4) squares in the input grid.
    2. Create a 3xN output grid where N = yellow_count * 3.
    3. Fill the output grid with a repeating pattern:
       - Top row: [0, 0, 8] repeated
       - Middle and bottom rows: [8, 8, 0] repeated
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Count yellow squares
    yellow_count = sum(row.count(4) for row in input_grid.values)
    
    # Calculate output width
    output_width = yellow_count * 3
    
    # Create empty output grid
    output = [[0 for _ in range(output_width)] for _ in range(3)]
    
    # Fill the output grid with the repeating pattern
    for i in range(output_width):
        if i % 3 == 2:
            output[0][i] = 8  # Top row
        if i % 3 == 0 or i % 3 == 1:
            output[1][i] = 8  # Middle row
            output[2][i] = 8  # Bottom row
    
    return ColoredGrid(values=output)
