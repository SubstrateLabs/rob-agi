from rob_agi.colored_grid import ColoredGrid

def solve_94414823(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by identifying two colors from the input grid and using them to fill
    the interior of a gray frame with a specific pattern.
    
    1. Identifies a top color from the second row and a bottom color from the second-to-last row.
    2. Creates a deep copy of the input grid.
    3. Fills the interior of the gray frame with a 2x2 pattern using the identified colors:
       - Top-left and bottom-right quadrants use the bottom color
       - Top-right and bottom-left quadrants use the top color
    
    Returns the modified grid with the interior of the frame filled according to the pattern.
    """
    # Find top color
    top_color = next((cell for cell in input_grid[1] if cell not in [0, 5]), None)
    
    # Find bottom color
    bottom_color = next((cell for cell in input_grid[-2] if cell not in [0, 5]), None)
    
    # Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Fill the interior of the frame
    for r in range(3, 7):
        for c in range(3, 7):
            quad_row = (r - 3) // 2
            quad_col = (c - 3) // 2
            if quad_row == quad_col:
                output_grid[r][c] = bottom_color
            else:
                output_grid[r][c] = top_color
    
    return output_grid
