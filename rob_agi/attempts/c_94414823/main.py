from rob_agi.colored_grid import ColoredGrid

def solve_94414823(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by identifying two colors from the perimeter of the gray frame
    and using them to fill the interior with a specific pattern.
    
    1. Scans the perimeter of the frame clockwise to find the first two distinct colors.
    2. Creates a deep copy of the input grid.
    3. Fills the interior of the gray frame with a 2x2 pattern using the identified colors:
       - Top-left and bottom-right quadrants use the first color found
       - Top-right and bottom-left quadrants use the second color found
    
    Returns the modified grid with the interior of the frame filled according to the pattern.
    """
    def find_colors(grid):
        first_color = second_color = None
        
        # Scan perimeter clockwise
        for r, c in ([(1, c) for c in range(1, 9)] +  # Top row
                     [(r, 8) for r in range(2, 8)] +  # Right column
                     [(8, c) for c in range(8, 0, -1)] +  # Bottom row
                     [(r, 1) for r in range(7, 1, -1)]):  # Left column
            color = grid[r][c]
            if color not in [0, 5]:
                if first_color is None:
                    first_color = color
                elif color != first_color:
                    second_color = color
                    return first_color, second_color
        
        return first_color, second_color

    first_color, second_color = find_colors(input_grid)
    
    # Create a deep copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Fill the interior of the frame
    for r in range(3, 7):
        for c in range(3, 7):
            if (r < 5) == (c < 5):
                output_grid[r][c] = first_color
            else:
                output_grid[r][c] = second_color
    
    return output_grid
