from rob_agi.colored_grid import ColoredGrid

def solve_bf32578f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid pattern into a centered shape.
    
    The function identifies the non-zero color in the input grid,
    determines whether to create a square or diamond shape,
    and places it in the center of a new grid of the same size
    as the input. The shape size is 4x4 for grids 10x10 or larger,
    and 3x3 for smaller grids.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with the transformed pattern.
    """
    def analyze_grid(grid):
        color = next(cell for row in grid.values for cell in row if cell != 0)
        non_zero_cells = [(r, c) for r, row in enumerate(grid.values) 
                          for c, val in enumerate(row) if val != 0]
        min_r, max_r = min(r for r, _ in non_zero_cells), max(r for r, _ in non_zero_cells)
        min_c, max_c = min(c for _, c in non_zero_cells), max(c for _, c in non_zero_cells)
        core_r, core_c = (min_r + max_r) // 2, (min_c + max_c) // 2
        core = [grid.values[r][c] for r in range(core_r, core_r+2) 
                for c in range(core_c, core_c+2)]
        return color, core

    def determine_shape(core, grid_size):
        is_square = all(core)
        size = 4 if min(grid_size) >= 10 else 3
        return 'square' if is_square else 'diamond', size

    def generate_shape(shape_type, size, color):
        if shape_type == 'square':
            return [[color] * size for _ in range(size)]
        else:  # diamond
            shape = [[0] * size for _ in range(size)]
            mid = size // 2
            for r in range(size):
                for c in range(size):
                    if size == 3:
                        if r == mid or c == mid:
                            shape[r][c] = color
                    else:  # size == 4
                        if r in (0, size-1) and c in (0, size-1):
                            continue
                        shape[r][c] = color
            return shape

    def place_shape(grid, shape):
        rows, cols = grid.get_dimensions()
        shape_size = len(shape)
        top = (rows - shape_size) // 2
        left = (cols - shape_size) // 2
        for r in range(shape_size):
            for c in range(shape_size):
                if 0 <= top + r < rows and 0 <= left + c < cols:
                    grid.values[top + r][left + c] = shape[r][c]
        return grid

    color, core = analyze_grid(input_grid)
    shape_type, size = determine_shape(core, input_grid.get_dimensions())
    shape = generate_shape(shape_type, size, color)
    output_grid = ColoredGrid(values=[[0] * len(row) for row in input_grid.values])
    return place_shape(output_grid, shape)
