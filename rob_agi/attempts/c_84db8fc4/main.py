from rob_agi.colored_grid import ColoredGrid

def solve_84db8fc4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding a continuous path of black squares
    connecting edges or corners, turning this path gray, changing other black
    squares to red, and leaving all other colored squares unchanged.

    1. Find a continuous path of black (0) squares connecting opposite sides or corners.
    2. Turn this path gray (5).
    3. Turn all other black squares red (2).
    4. Leave all other colored squares unchanged.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    # Find all black squares
    black_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 0]

    # Helper functions
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def is_black(r, c):
        return input_grid.values[r][c] == 0

    def is_edge(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    # DFS to find path
    def dfs(r, c, visited, path):
        if not is_valid(r, c) or not is_black(r, c) or (r, c) in visited:
            return None
        
        new_path = path + [(r, c)]
        visited.add((r, c))
        
        if is_edge(r, c) and len(new_path) > 1:
            return new_path
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            result = dfs(r + dr, c + dc, visited, new_path)
            if result:
                return result
        
        return None

    # Find gray path
    gray_path = None
    for r, c in black_squares:
        if is_edge(r, c):
            gray_path = dfs(r, c, set(), [])
            if gray_path:
                break

    # Transform grid
    if gray_path:
        for r, c in gray_path:
            output_grid.values[r][c] = 5  # Gray

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0 and (r, c) not in (gray_path or []):
                output_grid.values[r][c] = 2  # Red

    return output_grid
