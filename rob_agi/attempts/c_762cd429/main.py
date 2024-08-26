from rob_agi.colored_grid import ColoredGrid

def solve_762cd429(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding a 2x2 color cluster in the bottom-left corner
    into a larger pattern that fills most of the grid, leaving a 1-cell black border.
    
    1. Extracts the 2x2 color cluster from the bottom-left corner.
    2. Calculates the size of each color's square region based on grid dimensions.
    3. Creates a 2x2 pattern of larger squares using the input cluster's colors.
    4. Fills the inner area of a new grid by repeating the 2x2 pattern.
    5. Ensures a 1-cell black border around the pattern.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    input_values = input_grid.values
    height, width = len(input_values), len(input_values[0])
    
    # Extract 2x2 color cluster
    cluster = [
        [input_values[-2][-2], input_values[-2][-1]],
        [input_values[-1][-2], input_values[-1][-1]]
    ]
    
    # Calculate square size
    square_size = (max(width, height) - 2) // 2
    
    # Create 2x2 pattern
    pattern_2x2 = [[color * square_size for color in row] for row in cluster]
    pattern_2x2 = [row for row in pattern_2x2 for _ in range(square_size)]
    
    # Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill inner area with pattern
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            pattern_row = (row - 1) % (square_size * 2)
            pattern_col = (col - 1) % (square_size * 2)
            new_grid[row][col] = pattern_2x2[pattern_row][pattern_col]
    
    return ColoredGrid(values=new_grid)
