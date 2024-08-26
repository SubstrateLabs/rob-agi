from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells iteratively.
    
    The function identifies all sky blue cells and expands them to adjacent cells
    (including diagonals), replacing only black (0) and blue (1) cells. Red (2) cells
    and other colors act as barriers. The expansion continues until no further changes
    are possible.
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all initial sky blue cells
    3. Use a queue to process cells for expansion
    4. Expand sky blue cells iteratively, respecting barriers
    5. Return the modified grid
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Step 2: Identify all initial sky blue cells
    queue = deque(
        (r, c) for r in range(rows) for c in range(cols)
        if input_grid.get_cell(r, c) == 8
    )

    # Step 3 & 4: Process cells for expansion
    while queue:
        r, c = queue.popleft()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if new_grid.get_cell(nr, nc) in [0, 1]:  # black or blue
                        new_grid.set_cell(nr, nc, 8)
                        queue.append((nr, nc))

    # Step 5: Return the modified grid
    return new_grid
