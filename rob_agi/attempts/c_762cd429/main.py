from rob_agi.colored_grid import ColoredGrid

def solve_762cd429(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding a 2x2 color cluster in the bottom-left corner
    into a larger pattern that fills most of the grid.
    
    1. Extracts the 2x2 color cluster from the bottom-left corner.
    2. Determines the expansion factor (2x2 for smaller grids, 4x4 for larger grids).
    3. Calculates the size of each expanded square based on the smaller grid dimension.
    4. Creates an expanded pattern based on the original 2x2 cluster.
    5. Fills the new grid with the expanded pattern, maintaining the original grid size.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    input_values = input_grid.values
    height, width = len(input_values), len(input_values[0])
    
    # Extract 2x2 color cluster
    cluster = [
        [input_values[-2][-2], input_values[-2][-1]],
        [input_values[-1][-2], input_values[-1][-1]]
    ]
    
    # Determine expansion factor
    expansion_factor = 4 if min(height, width) > 14 else 2
    
    # Calculate square size
    square_size = min(width, height) // expansion_factor
    
    # Create new grid
    new_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Calculate starting position
    start_row = height - (expansion_factor * square_size)
    start_col = width - (expansion_factor * square_size)
    
    # Fill the grid with the expanded pattern
    for i in range(expansion_factor):
        for j in range(expansion_factor):
            if i < 2 and j < 2:
                color = cluster[i][j]
            elif i < 2:
                color = cluster[i][1]
            elif j < 2:
                color = cluster[1][j]
            else:
                color = cluster[0][0]
            
            for row in range(square_size):
                for col in range(square_size):
                    new_row = start_row + (i * square_size) + row
                    new_col = start_col + (j * square_size) + col
                    new_grid[new_row][new_col] = color
    
    return ColoredGrid(values=new_grid)
