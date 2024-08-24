from rob_agi.colored_grid import ColoredGrid

def solve_d90796e8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Adjacent green (3) and red (2) cells transform into sky blue (8) and black (0) respectively.
    2. The transformation occurs regardless of order (green-red or red-green).
    3. Only horizontal and vertical adjacencies are considered, not diagonal.
    4. The process is repeated until no further changes can be made.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    def transform_once(grid):
        height, width = grid.get_dimensions()
        new_grid = grid.deep_copy()
        changed = False
        
        for row in range(height):
            for col in range(width):
                cell = grid.get_cell(row, col)
                if cell not in [2, 3]:
                    continue
                
                # Check right neighbor
                if col + 1 < width:
                    right = grid.get_cell(row, col + 1)
                    if (cell == 2 and right == 3) or (cell == 3 and right == 2):
                        new_grid.set_cell(row, col, 8 if cell == 3 else 0)
                        new_grid.set_cell(row, col + 1, 0 if cell == 3 else 8)
                        changed = True
                
                # Check bottom neighbor
                if row + 1 < height:
                    bottom = grid.get_cell(row + 1, col)
                    if (cell == 2 and bottom == 3) or (cell == 3 and bottom == 2):
                        new_grid.set_cell(row, col, 8 if cell == 3 else 0)
                        new_grid.set_cell(row + 1, col, 0 if cell == 3 else 8)
                        changed = True
        
        return new_grid, changed

    result = input_grid
    while True:
        result, changed = transform_once(result)
        if not changed:
            break
    
    return result
