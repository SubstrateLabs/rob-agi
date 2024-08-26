from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating an inner diagonal skeleton of sky blue (8) regions with red (2).
    
    This function applies the following rules:
    1. Black cells (0) diagonally adjacent to sky blue cells (8) become red (2),
       but only if they are not orthogonally adjacent to any sky blue cells.
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
    
    # First pass: Mark potential red cells
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0:  # If the cell is black
                for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:  # Check diagonals
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 8:
                        output_grid.values[r][c] = -1  # Mark as potential red
                        break

    # Second pass: Confirm red cells and clean up
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == -1:  # If it's a potential red cell
                is_red = True
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:  # Check orthogonal neighbors
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 8:
                        is_red = False
                        break
                output_grid.values[r][c] = 2 if is_red else 0

    return output_grid
