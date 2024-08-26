from rob_agi.colored_grid import ColoredGrid

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells to their surrounding area.
    
    The function identifies all sky blue cells and expands them to a 3x3 square around
    each cell, replacing only black (0), blue (1), or yellow (4) cells. Red (2) cells
    and other colors are preserved.
    
    Steps:
    1. Create a deep copy of the input grid
    2. Scan the input grid for sky blue cells
    3. For each sky blue cell, expand it to a 3x3 square, respecting grid boundaries
       and only replacing specific colors
    4. Return the modified grid
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Step 2: Scan for sky blue cells
    sky_blue_cells = [
        (r, c) for r in range(rows) for c in range(cols)
        if input_grid.get_cell(r, c) == 8
    ]

    # Step 3: Process each sky blue cell
    for center_r, center_c in sky_blue_cells:
        # Define the 3x3 square
        for r in range(center_r - 1, center_r + 2):
            for c in range(center_c - 1, center_c + 2):
                # Check if within grid boundaries
                if 0 <= r < rows and 0 <= c < cols:
                    # Check cell color and update if necessary
                    cell_color = new_grid.get_cell(r, c)
                    if cell_color in [0, 1, 4]:  # black, blue, or yellow
                        new_grid.set_cell(r, c, 8)  # set to sky blue

    # Step 4: Return the modified grid
    return new_grid
