from rob_agi.colored_grid import ColoredGrid

def solve_58e15b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating diamond patterns from colored squares.
    
    The algorithm works as follows:
    1. Identifies all non-black squares in the input grid.
    2. For each color group, determines the maximum extent of its diamond pattern.
    3. Creates diamond patterns for each original colored square, extending to the determined extent.
    4. Resolves conflicts between overlapping patterns, using magenta (6) for edge intersections.
    5. Preserves the original colored squares from the input.
    6. Fills the remaining space with black (0).
    
    Returns a new ColoredGrid with the transformed diamond pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def find_colored_squares():
        colored_squares = {}
        for r in range(rows):
            for c in range(cols):
                color = input_grid.values[r][c]
                if color != 0:
                    if color not in colored_squares:
                        colored_squares[color] = []
                    colored_squares[color].append((r, c))
        return colored_squares
    
    def calculate_max_extent(squares):
        if not squares:
            return 0
        max_r = max(r for r, _ in squares)
        min_r = min(r for r, _ in squares)
        max_c = max(c for _, c in squares)
        min_c = min(c for _, c in squares)
        return max(max_r - min_r, max_c - min_c)
    
    def manhattan_distance(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)
    
    def draw_diamond(r, c, color, max_extent):
        for nr in range(rows):
            for nc in range(cols):
                distance = manhattan_distance(r, c, nr, nc)
                if distance <= max_extent:
                    if output_grid.values[nr][nc] == 0:
                        output_grid.values[nr][nc] = color
                    elif output_grid.values[nr][nc] != color:
                        if distance == max_extent:
                            output_grid.values[nr][nc] = 6  # Intersection
    
    colored_squares = find_colored_squares()
    max_extents = {color: calculate_max_extent(squares) for color, squares in colored_squares.items()}
    
    # Draw diamonds for each color
    for color, squares in colored_squares.items():
        for r, c in squares:
            draw_diamond(r, c, color, max_extents[color])
    
    # Preserve original squares
    for color, squares in colored_squares.items():
        for r, c in squares:
            output_grid.values[r][c] = color
    
    return output_grid
