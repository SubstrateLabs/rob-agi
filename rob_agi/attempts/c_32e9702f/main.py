from rob_agi.colored_grid import ColoredGrid

def solve_32e9702f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Preserve non-black regions.
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
                # Preserve non-black color
                output_grid.values[r][c] = input_grid.values[r][c]
                
                # Expand downwards
                for i in range(r+1, rows):
                    if input_grid.values[i][c] != 0:
                        break
                    output_grid.values[i][c] = 5
                
                # Expand to the right
                for j in range(c+1, cols):
                    if input_grid.values[r][j] != 0:
                        break
                    output_grid.values[r][j] = 5

    return output_grid
