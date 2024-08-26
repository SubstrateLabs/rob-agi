from rob_agi.colored_grid import ColoredGrid

def solve_dd2401ed(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving the gray vertical line to the right
    and adjusting the positions of colored dots.
    
    1. Finds the original gray line position.
    2. Calculates the new gray line position (double the original).
    3. Creates a new grid with the gray line in the new position.
    4. Copies colored dots:
       - Dots to the left of the original gray line remain in their absolute positions.
       - Dots to the right of the original gray line maintain their relative positions
         to the new gray line, potentially being pushed off the grid if out of bounds.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the original gray line position
    original_pos = next(i for i, color in enumerate(input_grid.values[0]) if color == 5)
    
    # Calculate the new gray line position
    new_pos = original_pos * 2
    
    # Create a new grid
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the gray line to the new position
    for row in range(rows):
        new_grid.values[row][new_pos] = 5
    
    # Process the colored dots
    for row in range(rows):
        for col in range(cols):
            color = input_grid.values[row][col]
            if color != 0 and color != 5:
                if col < original_pos:
                    new_grid.values[row][col] = color
                else:
                    new_col = new_pos + (col - original_pos)
                    if new_col < cols:
                        new_grid.values[row][new_col] = color
    
    return new_grid
