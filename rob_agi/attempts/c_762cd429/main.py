from rob_agi.colored_grid import ColoredGrid

def solve_762cd429(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding a 2x2 color cluster in the bottom-left corner
    into a larger pattern that fills most of the grid, leaving a 1-cell black border.
    
    1. Extracts the 2x2 color cluster from the bottom-left corner.
    2. Calculates the size of each expanded square based on the smaller grid dimension.
    3. Creates an expanded 2x2 pattern, repeating the top-left color in the top-right.
    4. Fills the bottom-right area of a new grid with the expanded pattern.
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
    square_size = (min(width, height) - 2) // 2
    
    # Create expanded 2x2 pattern
    expanded_pattern = [
        [cluster[0][0], cluster[0][1]],
        [cluster[1][0], cluster[1][1]]
    ]
    
    # Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Calculate starting position
    start_row = height - (2 * square_size) - 1
    start_col = width - (2 * square_size) - 1
    
    # Fill bottom-right area with expanded pattern
    for i in range(2):
        for j in range(2):
            color = expanded_pattern[i][j]
            for row in range(square_size):
                for col in range(square_size):
                    new_row = start_row + (i * square_size) + row
                    new_col = start_col + (j * square_size) + col
                    new_grid[new_row][new_col] = color
    
    # Special case: repeat top-left color in top-right
    for row in range(start_row, start_row + square_size):
        for col in range(start_col + square_size, start_col + 2 * square_size):
            new_grid[row][col] = expanded_pattern[0][0]
    
    return ColoredGrid(values=new_grid)
