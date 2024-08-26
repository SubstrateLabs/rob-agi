from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells using a flood-fill algorithm.
    
    The function identifies all sky blue cells and expands them to adjacent cells,
    replacing only black (0) and blue (1) cells. Red (2) cells and other colors act as barriers.
    The expansion continues until it reaches the edge of the grid or a barrier color.
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all initial sky blue cells
    3. Apply a flood-fill algorithm from each sky blue cell, expanding to adjacent cells
    4. Return the modified grid
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    def flood_fill(grid, start_r, start_c):
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cell_color = grid.get_cell(nr, nc)
                    if cell_color in [0, 1]:  # black or blue
                        grid.set_cell(nr, nc, 8)  # set to sky blue
                        queue.append((nr, nc))

    # Step 1: Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Step 2: Identify all initial sky blue cells
    sky_blue_cells = [
        (r, c) for r in range(rows) for c in range(cols)
        if input_grid.get_cell(r, c) == 8
    ]

    # Step 3: Apply flood-fill algorithm from each sky blue cell
    for r, c in sky_blue_cells:
        flood_fill(new_grid, r, c)

    # Step 4: Return the modified grid
    return new_grid
