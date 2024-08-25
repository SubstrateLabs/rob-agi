from rob_agi.colored_grid import ColoredGrid

def solve_58e15b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating diagonal lines from colored squares.
    
    The algorithm works as follows:
    1. Identifies colored squares in the input grid.
    2. Determines the outer and inner colors based on square positions.
    3. Generates diagonal lines for the outer color, extending to grid edges or other colored squares.
    4. Generates diagonal lines for the inner color, stopping at intersections with outer color lines.
    5. Handles intersections by coloring them magenta (6).
    6. Preserves the original colored squares from the input.
    7. Fills the remaining space with black (0).
    
    Returns a new ColoredGrid with the transformed diagonal line pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def find_colored_squares():
        sky_squares = []
        green_squares = []
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == 8:
                    sky_squares.append((r, c))
                elif input_grid.values[r][c] == 3:
                    green_squares.append((r, c))
        return sky_squares, green_squares
    
    def is_outer_color(sky_squares, green_squares):
        sky_dist = min(min(r, c, rows-1-r, cols-1-c) for r, c in sky_squares)
        green_dist = min(min(r, c, rows-1-r, cols-1-c) for r, c in green_squares)
        return 8 if sky_dist < green_dist else 3
    
    def draw_diagonal_lines(squares, color, is_outer):
        for r, c in squares:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols:
                    if output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = color
                    elif output_grid.values[nr][nc] != color:
                        if not is_outer:
                            output_grid.values[nr][nc] = 6  # Intersection
                        break
                    if not is_outer and output_grid.values[nr][nc] == 6:
                        break
                    nr, nc = nr + dr, nc + dc
    
    sky_squares, green_squares = find_colored_squares()
    outer_color = is_outer_color(sky_squares, green_squares)
    inner_color = 3 if outer_color == 8 else 8
    
    draw_diagonal_lines(sky_squares if outer_color == 8 else green_squares, outer_color, True)
    draw_diagonal_lines(green_squares if outer_color == 8 else sky_squares, inner_color, False)
    
    # Preserve original squares
    for squares in [sky_squares, green_squares]:
        for r, c in squares:
            output_grid.values[r][c] = input_grid.values[r][c]
    
    return output_grid
