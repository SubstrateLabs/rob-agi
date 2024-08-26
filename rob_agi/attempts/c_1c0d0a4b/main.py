from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by marking the boundaries of sky blue (8) regions with red (2).
    
    This function identifies cells that should be marked red based on the following rule:
    - A black cell (0) becomes red (2) if it is adjacent (including diagonally) to a sky blue cell (8).
    - Sky blue cells (8) remain unchanged.
    - All other cells become black (0).
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings at the boundaries of sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def is_adjacent_to_sky_blue(row: int, col: int) -> bool:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.values[nr][nc] == 8:
                        return True
        return False
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                output_grid.values[r][c] = 8
            elif input_grid.values[r][c] == 0 and is_adjacent_to_sky_blue(r, c):
                output_grid.values[r][c] = 2
    
    return output_grid
