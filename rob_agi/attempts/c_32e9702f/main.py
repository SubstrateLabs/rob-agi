from rob_agi.colored_grid import ColoredGrid

def solve_32e9702f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Preserve non-black regions, except when they are in the expansion path of another non-black region.
    2. Expand non-black regions downwards and to the right with gray (5) cells.
    3. Fill all remaining cells with gray (5).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[5 for _ in range(cols)] for _ in range(rows)])

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                # Check if this cell is in the expansion path of a previous non-black cell
                if r > 0 and input_grid.values[r-1][c] != 0:
                    continue
                if c > 0 and input_grid.values[r][c-1] != 0:
                    continue
                
                # Preserve non-black color
                output_grid.values[r][c] = input_grid.values[r][c]
                
                # Expand downwards
                for i in range(r+1, rows):
                    output_grid.values[i][c] = 5
                    if input_grid.values[i][c] != 0:
                        break
                
                # Expand to the right
                for j in range(c+1, cols):
                    output_grid.values[r][j] = 5
                    if input_grid.values[r][j] != 0:
                        break

    return output_grid
