from rob_agi.colored_grid import ColoredGrid

def solve_423a55dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by shifting colored cells diagonally up-left and filling in a diagonal pattern.
    The transformation creates an illusion of 45-degree rotation for larger shapes,
    while correctly handling smaller shapes or those close to the edges.
    
    1. Shift colored cells up-left until they reach the left edge of the grid.
    2. Fill in diagonal cells to create a staircase-like effect.
    3. Preserve the original shape if it's already at the edge.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def is_valid_position(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_color(r, c):
        return input_grid.values[r][c] if is_valid_position(r, c) else 0

    # Find the leftmost column with non-zero cells
    left_edge = min((c for r in range(rows) for c in range(cols) if input_grid.values[r][c] != 0), default=0)
    
    # Number of shifts needed
    shifts = left_edge

    for _ in range(shifts):
        new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
        for r in range(rows):
            for c in range(cols):
                color = get_color(r, c)
                if color != 0:
                    # Move up-left
                    new_r, new_c = r - 1, c - 1
                    if is_valid_position(new_r, new_c):
                        new_grid.values[new_r][new_c] = color
                    else:
                        new_grid.values[r][c] = color
                    
                    # Fill diagonal down-right
                    if is_valid_position(r, c):
                        new_grid.values[r][c] = color
        
        output_grid = new_grid

    # Preserve original shape if it's already at the edge
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and output_grid.values[r][c] == 0:
                output_grid.values[r][c] = input_grid.values[r][c]

    return output_grid
