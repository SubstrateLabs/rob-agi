from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by marking the boundaries between sky blue (8) regions and black (0) spaces with red (2).
    
    This function identifies cells that should be marked red based on the following rules:
    1. A black cell with at least one sky blue neighbor and at least one black neighbor becomes red.
    2. A black cell with exactly two diagonal sky blue neighbors that are not directly adjacent becomes red.
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings at the boundaries between sky blue and black regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def should_be_red(row: int, col: int) -> bool:
        if input_grid.values[row][col] != 0:
            return False
        
        sky_blue_count = 0
        black_count = 0
        diagonal_sky_blue = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if input_grid.values[nr][nc] == 8:
                        sky_blue_count += 1
                        if dr != 0 and dc != 0:
                            diagonal_sky_blue.append((nr, nc))
                    elif input_grid.values[nr][nc] == 0:
                        black_count += 1
        
        if sky_blue_count > 0 and black_count > 0:
            return True
        
        if len(diagonal_sky_blue) == 2:
            r1, c1 = diagonal_sky_blue[0]
            r2, c2 = diagonal_sky_blue[1]
            if abs(r1 - r2) == 2 and abs(c1 - c2) == 2:
                return True
        
        return False
    
    for r in range(rows):
        for c in range(cols):
            if should_be_red(r, c):
                output_grid.values[r][c] = 2
    
    return output_grid
