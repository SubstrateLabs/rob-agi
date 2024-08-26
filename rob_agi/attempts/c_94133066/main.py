from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting all non-black squares and creating a new, compact grid.
    
    1. Scans the input grid to find all non-black squares.
    2. Determines the smallest rectangle that contains all non-black squares.
    3. Creates a new grid filled with blue (1) of the determined size.
    4. Places all non-black squares from the input grid into the new grid, maintaining relative positions.
    5. Rotates the grid if it results in a more compact representation (width > height).
    6. Ensures the final grid is square by adding blue (1) columns or rows if necessary.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    non_black = [(r, c, val) for r, row in enumerate(input_grid.values) 
                 for c, val in enumerate(row) if val != 0]
    
    if not non_black:
        return ColoredGrid(values=[[1]])
    
    min_r = min(r for r, _, _ in non_black)
    max_r = max(r for r, _, _ in non_black)
    min_c = min(c for _, c, _ in non_black)
    max_c = max(c for _, c, _ in non_black)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    new_grid = [[1 for _ in range(width)] for _ in range(height)]
    
    for r, c, val in non_black:
        new_grid[r - min_r][c - min_c] = val
    
    if width > height:
        new_grid = list(zip(*reversed(new_grid)))
        width, height = height, width
    
    max_dim = max(width, height)
    square_grid = [[1 for _ in range(max_dim)] for _ in range(max_dim)]
    
    for r in range(height):
        for c in range(width):
            square_grid[r][c] = new_grid[r][c]
    
    return ColoredGrid(values=square_grid)
