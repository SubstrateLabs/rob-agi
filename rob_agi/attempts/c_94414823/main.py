from rob_agi.colored_grid import ColoredGrid

def solve_94414823(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by identifying two colors from the perimeter of the gray frame
    and using them to fill the interior with a specific pattern.
    
    1. Scans the entire perimeter of the frame clockwise to find the first two distinct colors and their positions.
    2. Determines which color comes first when moving clockwise from the top-left corner.
    3. Creates a deep copy of the input grid.
    4. Fills the interior of the gray frame with a 2x2 pattern using the identified colors:
       - Top-left and bottom-right quadrants use the color that comes first clockwise
       - Top-right and bottom-left quadrants use the other color
    
    Returns the modified grid with the interior of the frame filled according to the pattern.
    """
    def find_colors_and_positions():
        colors = []
        positions = []
        # Scan top row
        for c in range(1, 9):
            if input_grid[1][c] not in [0, 5] and input_grid[1][c] not in colors:
                colors.append(input_grid[1][c])
                positions.append((1, c))
                if len(colors) == 2:
                    return colors, positions
        # Scan right column
        for r in range(2, 9):
            if input_grid[r][8] not in [0, 5] and input_grid[r][8] not in colors:
                colors.append(input_grid[r][8])
                positions.append((r, 8))
                if len(colors) == 2:
                    return colors, positions
        # Scan bottom row
        for c in range(8, 0, -1):
            if input_grid[8][c] not in [0, 5] and input_grid[8][c] not in colors:
                colors.append(input_grid[8][c])
                positions.append((8, c))
                if len(colors) == 2:
                    return colors, positions
        # Scan left column
        for r in range(7, 1, -1):
            if input_grid[r][1] not in [0, 5] and input_grid[r][1] not in colors:
                colors.append(input_grid[r][1])
                positions.append((r, 1))
                if len(colors) == 2:
                    return colors, positions

    colors, positions = find_colors_and_positions()
    
    # Determine which color comes first clockwise
    if positions[0][0] < positions[1][0] or (positions[0][0] == positions[1][0] and positions[0][1] < positions[1][1]):
        first_color, second_color = colors
    else:
        second_color, first_color = colors

    output_grid = input_grid.deep_copy()

    # Fill the interior
    for r in range(3, 7):
        for c in range(3, 7):
            if (r < 5) == (c < 5):
                output_grid[r][c] = first_color
            else:
                output_grid[r][c] = second_color

    return output_grid
