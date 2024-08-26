from rob_agi.colored_grid import ColoredGrid

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells in a 3x3 pattern.
    
    The function identifies all sky blue cells and expands them to adjacent cells
    in a 3x3 square, replacing only black (0) and blue (1) cells. Red (2) cells
    and other colors act as barriers. The expansion is simulated to occur simultaneously.
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all initial sky blue cells
    3. Calculate cells to be changed based on 3x3 expansion
    4. Apply changes to create the final grid
    5. Return the modified grid
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Step 2: Identify all initial sky blue cells
    sky_blue_cells = [
        (r, c) for r in range(rows) for c in range(cols)
        if input_grid.get_cell(r, c) == 8
    ]

    # Step 3: Calculate cells to be changed
    cells_to_change = set()
    for r, c in sky_blue_cells:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.get_cell(nr, nc) in [0, 1]:  # black or blue
                        cells_to_change.add((nr, nc))

    # Step 4: Apply changes
    for r, c in cells_to_change:
        new_grid.set_cell(r, c, 8)

    # Step 5: Return the modified grid
    return new_grid
