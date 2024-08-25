from rob_agi.colored_grid import ColoredGrid

def solve_58e15b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating nested 'U' shapes based on the colored squares.
    
    The algorithm works as follows:
    1. Identifies the outermost and innermost colored squares.
    2. Creates an outer 'U' shape using the color of the outermost squares.
    3. Creates an inner 'U' shape using the color of the innermost squares.
    4. Handles intersections by coloring them magenta (6).
    5. Preserves the original colored squares from the input.
    6. Fills the remaining space with black (0).
    
    Returns a new ColoredGrid with the transformed nested 'U' shape pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def find_colored_squares():
        colored = []
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != 0:
                    colored.append((r, c, input_grid.values[r][c]))
        return colored
    
    def create_u_shape(color, is_outer):
        for r in range(rows):
            for c in range(cols):
                if is_outer:
                    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                        output_grid.values[r][c] = color
                else:
                    if (r == 1 or r == rows - 2) and 1 <= c < cols - 1:
                        output_grid.values[r][c] = color
                    elif (c == 1 or c == cols - 2) and 1 < r < rows - 1:
                        output_grid.values[r][c] = color
    
    colored_squares = find_colored_squares()
    outer_color = colored_squares[0][2]
    inner_color = colored_squares[-1][2]
    
    create_u_shape(outer_color, True)
    create_u_shape(inner_color, False)
    
    # Handle intersections
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == outer_color and input_grid.values[r][c] == inner_color:
                output_grid.values[r][c] = 6
    
    # Preserve original squares
    for r, c, color in colored_squares:
        output_grid.values[r][c] = color
    
    return output_grid
