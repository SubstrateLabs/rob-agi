from rob_agi.colored_grid import ColoredGrid

def solve_d37a1ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding the red frame inwards while preserving other colored cells.
    
    The function:
    1. Identifies the original red frame
    2. Expands the frame inwards as much as possible without overlapping non-black cells
    3. Preserves the original black border, top and bottom frame rows, and all non-red, non-black cells
    4. Preserves black cells adjacent to non-red, non-black cells
    5. Returns a new grid with the expanded red frame
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find original frame boundaries
    top = next(r for r in range(rows) if 2 in input_grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 2 in input_grid.values[r])
    left = next(c for c in range(cols) if any(row[c] == 2 for row in input_grid.values))
    right = next(c for c in range(cols-1, -1, -1) if any(row[c] == 2 for row in input_grid.values))

    # Copy black border
    new_grid.values[0] = input_grid.values[0]
    new_grid.values[-1] = input_grid.values[-1]
    for r in range(rows):
        new_grid.values[r][0] = input_grid.values[r][0]
        new_grid.values[r][-1] = input_grid.values[r][-1]

    # Copy top and bottom frame rows
    new_grid.values[top] = input_grid.values[top]
    new_grid.values[bottom] = input_grid.values[bottom]

    # Process middle rows
    for r in range(top + 1, bottom):
        left_ptr = left
        right_ptr = right
        
        # Expand inwards
        while left_ptr < right_ptr:
            if input_grid.values[r][left_ptr + 1] == 0:
                left_ptr += 1
            elif input_grid.values[r][right_ptr - 1] == 0:
                right_ptr -= 1
            else:
                break

        # Fill expanded frame
        for c in range(left, left_ptr + 1):
            new_grid.values[r][c] = 2
        for c in range(right_ptr, right + 1):
            new_grid.values[r][c] = 2

        # Copy non-red, non-black cells and adjacent black cells
        for c in range(left, right + 1):
            if input_grid.values[r][c] not in [0, 2]:
                new_grid.values[r][c] = input_grid.values[r][c]
                # Copy adjacent black cells
                if c > left and input_grid.values[r][c-1] == 0:
                    new_grid.values[r][c-1] = 0
                if c < right and input_grid.values[r][c+1] == 0:
                    new_grid.values[r][c+1] = 0

        # Fill remaining cells inside the original frame with red
        for c in range(left_ptr + 1, right_ptr):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = 2

    return new_grid
