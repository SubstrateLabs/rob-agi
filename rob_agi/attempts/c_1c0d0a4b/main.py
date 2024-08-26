from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by marking the boundaries of sky blue (8) regions with red (2).
    
    This function applies the following rules:
    1. Black cells (0) orthogonally adjacent to sky blue cells (8) become red (2).
    2. All sky blue cells (8) become black (0) in the output.
    3. All other cells remain black (0).
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions on a black background.
    
    Returns:
    ColoredGrid: A new grid with red markings at the orthogonal boundaries of sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # First pass: Mark red boundaries
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = 2
    
    return output_grid
