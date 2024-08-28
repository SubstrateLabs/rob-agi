from rob_agi.colored_grid import ColoredGrid

def solve_dd2401ed(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving the gray vertical line to the right
    and adjusting the positions of colored dots.
    
    1. Finds the original gray line position.
    2. Calculates the new gray line position (original position + original position).
    3. Creates a new grid with the same dimensions as the input grid.
    4. Processes colored dots:
       - Blue (1) dots remain in their original positions.
       - Red (2) dots are shifted left by the same amount as the gray line moved right,
         wrapping around to the right side if necessary.
    5. Places the gray line in its new position.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the original gray line position
    original_pos = next(i for i, color in enumerate(input_grid.values[0]) if color == 5)
    
    # Calculate the new gray line position
    new_pos = min((original_pos * 2) % cols, cols - 1)
    shift = new_pos - original_pos
    
    # Create a new grid
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Process the input grid
    for row in range(rows):
        for col in range(cols):
            color = input_grid.values[row][col]
            if color == 1:
                # Blue dots remain in their original positions
                new_grid.values[row][col] = color
            elif color == 2:
                # Shift red dots left, wrapping around if necessary
                new_col = (col - shift) % cols
                new_grid.values[row][new_col] = color
        
        # Place the gray line in its new position
        new_grid.values[row][new_pos] = 5
    
    return new_grid
