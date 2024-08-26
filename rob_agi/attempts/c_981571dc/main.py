from rob_agi.colored_grid import ColoredGrid

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function extends existing colors from the edges inward, replacing black (0) cells
    with appropriate colors based on their non-black neighbors. This process continues
    until no black cells remain or no further changes can be made.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    def has_black_neighbors(grid, row, col):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < len(grid.values) and 0 <= new_col < len(grid.values[0]):
                if grid.values[new_row][new_col] == 0:
                    return True
        return False

    def get_replacement_color(grid, row, col):
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # left, up, right, down
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < len(grid.values) and 0 <= new_col < len(grid.values[0]):
                if grid.values[new_row][new_col] != 0:
                    return grid.values[new_row][new_col]
        return 0  # This should never happen if used correctly

    grid = input_grid.deep_copy()
    changed = True
    while changed:
        changed = False
        for row in range(len(grid.values)):
            for col in range(len(grid.values[0])):
                if grid.values[row][col] == 0 and not has_black_neighbors(grid, row, col):
                    grid.values[row][col] = get_replacement_color(grid, row, col)
                    changed = True

    return grid
