from rob_agi.colored_grid import ColoredGrid

def solve_d37a1ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding the red frame inwards while preserving gray cells and their borders.
    
    The function:
    1. Identifies the original red frame
    2. Expands the side edges inward up to 2 cells, respecting gray cells
    3. Fills in the inner area, creating a consistent expanded frame
    4. Ensures all gray cells maintain a 1-cell black border
    5. Processes top and bottom rows to maintain original thickness
    6. Returns a new grid with the expanded red frame
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    # Find original frame boundaries
    top = next(r for r in range(rows) if 2 in new_grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 2 in new_grid.values[r])
    left = next(c for c in range(cols) if any(row[c] == 2 for row in new_grid.values))
    right = next(c for c in range(cols-1, -1, -1) if any(row[c] == 2 for row in new_grid.values))

    # Process rows (excluding top and bottom frame rows)
    for r in range(top + 1, bottom):
        # Left expansion
        for c in range(left + 1, min(left + 3, (left + right) // 2)):
            if new_grid.values[r][c] == 0 and not is_adjacent_to_gray(new_grid, r, c):
                new_grid.values[r][c] = 2
            else:
                break
        # Right expansion
        for c in range(right - 1, max(right - 3, (left + right) // 2), -1):
            if new_grid.values[r][c] == 0 and not is_adjacent_to_gray(new_grid, r, c):
                new_grid.values[r][c] = 2
            else:
                break
        # Fill between expanded edges
        for c in range(left + 1, right):
            if new_grid.values[r][c] == 0 and not is_adjacent_to_gray(new_grid, r, c):
                new_grid.values[r][c] = 2

    # Process top and bottom rows
    for r in [top + 1, bottom - 1]:
        for c in range(left + 1, right):
            if new_grid.values[r][c] == 0 and not is_adjacent_to_gray(new_grid, r, c):
                new_grid.values[r][c] = 2

    # Preserve gray cell borders
    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] == 5:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and new_grid.values[nr][nc] == 2:
                            new_grid.values[nr][nc] = 0

    return new_grid

def is_adjacent_to_gray(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 5:
                return True
    return False
