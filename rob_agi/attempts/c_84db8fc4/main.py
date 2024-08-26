from rob_agi.colored_grid import ColoredGrid

def solve_84db8fc4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding the longest continuous path of black squares
    connecting any two edge points, turning this path gray, changing other black
    squares to red, and leaving all other colored squares unchanged.

    1. Find the longest continuous path of black (0) squares connecting any two edge points.
    2. Turn this path gray (5).
    3. Turn all other black squares red (2).
    4. Leave all other colored squares unchanged.

    If no valid path is found, all black squares are turned red.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and input_grid.values[r][c] == 0

    def is_edge(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def find_longest_path():
        longest_path = []
        edge_squares = [(r, c) for r in range(rows) for c in range(cols) 
                        if is_edge(r, c) and input_grid.values[r][c] == 0]

        for start_r, start_c in edge_squares:
            stack = [(start_r, start_c, [(start_r, start_c)], set([(start_r, start_c)]))]
            while stack:
                r, c, path, visited = stack.pop()
                if is_edge(r, c) and len(path) > 1 and len(path) > len(longest_path):
                    longest_path = path

                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc) and (nr, nc) not in visited:
                        new_path = path + [(nr, nc)]
                        new_visited = visited.copy()
                        new_visited.add((nr, nc))
                        stack.append((nr, nc, new_path, new_visited))

        return longest_path

    longest_path = find_longest_path()

    if longest_path:
        for r, c in longest_path:
            output_grid.values[r][c] = 5  # Gray

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0 and (r, c) not in longest_path:
                output_grid.values[r][c] = 2  # Red

    return output_grid
