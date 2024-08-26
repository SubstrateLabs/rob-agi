from rob_agi.colored_grid import ColoredGrid

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rule:
    For each 3x3 block in the grid, if any cell in the block contains red (2),
    fill the entire block with red (2), except for yellow (4) cells which remain unchanged.
    The grid is processed in 3x3 blocks, handling edge cases where grid dimensions
    are not multiples of 3.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    for i in range(0, rows, 3):
        for j in range(0, cols, 3):
            has_red = any(output_grid.values[x][y] == 2 
                          for x in range(i, min(i+3, rows)) 
                          for y in range(j, min(j+3, cols)))
            
            if has_red:
                for x in range(i, min(i+3, rows)):
                    for y in range(j, min(j+3, cols)):
                        if output_grid.values[x][y] != 4:
                            output_grid.values[x][y] = 2

    return output_grid
