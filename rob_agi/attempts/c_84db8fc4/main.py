from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_84db8fc4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding the longest continuous path of black squares
    connecting any two edge points on different edges, turning this path gray,
    and leaving all other squares unchanged.

    1. Find the longest continuous path of black (0) squares connecting any two edge points on different edges.
    2. Turn this path gray (5).
    3. Leave all other squares unchanged, including other black squares.

    If no valid path is found, the grid remains unchanged.

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

    def get_edge(r, c):
        if r == 0: return 'top'
        if r == rows - 1: return 'bottom'
        if c == 0: return 'left'
        if c == cols - 1: return 'right'
        return None

    def dfs(r, c, start_edge, path, visited):
        if not is_valid(r, c) or (r, c) in visited:
            return []
        
        current_path = path + [(r, c)]
        visited.add((r, c))
        
        end_edge = get_edge(r, c)
        if end_edge and end_edge != start_edge:
            return current_path
        
        longest_path = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            new_path = dfs(nr, nc, start_edge, current_path, visited.copy())
            if len(new_path) > len(longest_path):
                longest_path = new_path
        
        return longest_path

    longest_path = []
    for r in range(rows):
        for c in range(cols):
            if is_edge(r, c) and input_grid.values[r][c] == 0:
                start_edge = get_edge(r, c)
                path = dfs(r, c, start_edge, [], set())
                if len(path) > len(longest_path):
                    longest_path = path

    if longest_path:
        for r, c in longest_path:
            output_grid.values[r][c] = 5  # Gray

    return output_grid
