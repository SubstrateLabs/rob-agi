from rob_agi.colored_grid import ColoredGrid

def solve_dd2401ed(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving the gray vertical line to the right
    and adjusting the positions of colored dots.
    
    1. Finds the original gray line position.
    2. Calculates the new gray line position (original position + original position).
    3. Creates a new grid with the gray line in the new position.
    4. Copies colored dots:
       - Dots to the left of the original gray line remain in their absolute positions.
       - Dots to the right of the original gray line are shifted left by the same amount
         as the gray line moved right, potentially being removed if pushed off the left edge.
    5. Places the gray line in its new position.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the original gray line position
    original_pos = next(i for i, color in enumerate(input_grid.values[0]) if color == 5)
    
    # Calculate the new gray line position and shift amount
    new_pos = original_pos + original_pos
    shift = new_pos - original_pos
    
    # Create a new grid
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Process the input grid
    for row in range(rows):
        for col in range(cols):
            color = input_grid.values[row][col]
            if col < original_pos:
                # Copy colors to the left of the original gray line
                new_grid.values[row][col] = color
            elif col > original_pos and color != 0:
                # Shift colors to the right of the original gray line
                new_col = col - shift
                if new_col >= 0:
                    new_grid.values[row][new_col] = color
        
        # Place the gray line in its new position
        new_grid.values[row][new_pos] = 5
    
    return new_grid
