from rob_agi.colored_grid import ColoredGrid

def solve_d37a1ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding the red frame inwards while preserving other colored cells.
    
    The function:
    1. Identifies the original red frame
    2. Expands the frame inwards as much as possible without overlapping non-black cells
    3. Preserves the original black border and all non-red, non-black cells
    4. Returns a new grid with the expanded red frame
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find original frame boundaries
    top = next(r for r in range(rows) if 2 in input_grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 2 in input_grid.values[r])
    left = next(c for c in range(cols) if any(row[c] == 2 for row in input_grid.values))
    right = next(c for c in range(cols-1, -1, -1) if any(row[c] == 2 for row in input_grid.values))

    # Find non-black, non-red cells
    special_cells = [(r, c) for r in range(top, bottom+1) for c in range(left, right+1)
                     if input_grid.values[r][c] not in [0, 2]]

    # Calculate new frame boundaries
    new_top = max(top, min(r for r, _ in special_cells) - 1) if special_cells else bottom
    new_bottom = min(bottom, max(r for r, _ in special_cells) + 1) if special_cells else top
    new_left = max(left, min(c for _, c in special_cells) - 1) if special_cells else right
    new_right = min(right, max(c for _, c in special_cells) + 1) if special_cells else left

    # Fill expanded frame
    for r in range(new_top, new_bottom + 1):
        for c in range(new_left, new_right + 1):
            new_grid.values[r][c] = 2

    # Copy black border
    new_grid.values[0] = input_grid.values[0]
    new_grid.values[-1] = input_grid.values[-1]
    for r in range(rows):
        new_grid.values[r][0] = input_grid.values[r][0]
        new_grid.values[r][-1] = input_grid.values[r][-1]

    # Copy non-red, non-black cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] not in [0, 2]:
                new_grid.values[r][c] = input_grid.values[r][c]

    return new_grid
