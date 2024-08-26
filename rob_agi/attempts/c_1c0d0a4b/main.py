from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an inner diagonal skeleton of sky blue (8) regions with red (2).
    
    This function applies the following rules:
    1. Black cells (0) diagonally adjacent to sky blue cells (8) become red (2),
       but only if they are part of the inner boundary of a sky blue region.
    2. All sky blue cells (8) become black (0) in the output.
    3. All other cells remain black (0).
    
    The transformation creates a "skeleton" of the original sky blue regions,
    marking their inner diagonal boundaries with red.
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings representing the inner diagonal skeleton of the original sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def count_diagonal_sky_blue(r, c):
        count = 0
        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 8:
                count += 1
        return count
    
    # First pass: Mark potential red cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0:  # If the cell is black
                sky_blue_count = count_diagonal_sky_blue(r, c)
                if 0 < sky_blue_count < 4:  # At least one, but not all diagonal neighbors are sky blue
                    output_grid.values[r][c] = 2  # Mark as red
    
    return output_grid
