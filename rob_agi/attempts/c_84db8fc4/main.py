from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_84db8fc4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding the longest continuous path of black squares
    connecting any two edge points on different edges, turning this path gray,
    and changing all other black squares to red.

    1. Find the longest continuous path of black (0) squares connecting any two edge points on different edges.
    2. Turn this path gray (5).
    3. Change all other black squares to red (2).
    4. Leave all other colored squares unchanged.

    If no valid path is found, all black squares are changed to red.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and input_grid.values[r][c] == 0

    def is_edge(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def get_edge(r: int, c: int) -> str:
        if r == 0: return 'top'
        if r == rows - 1: return 'bottom'
        if c == 0: return 'left'
        if c == cols - 1: return 'right'
        return ''

    def dfs(r: int, c: int, start_edge: str, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not is_valid(r, c) or (r, c) in visited:
            return []
        
        visited.add((r, c))
        current_path = [(r, c)]
        
        end_edge = get_edge(r, c)
        if end_edge and end_edge != start_edge:
            return current_path
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            new_path = dfs(nr, nc, start_edge, visited.copy())
            if len(new_path) > 0:
                return current_path + new_path
        
        return []

    longest_path = []
    for r in range(rows):
        for c in range(cols):
            if is_edge(r, c) and input_grid.values[r][c] == 0:
                start_edge = get_edge(r, c)
                path = dfs(r, c, start_edge, set())
                if len(path) > len(longest_path):
                    longest_path = path

    longest_path_set = set(longest_path)

    for r in range(rows):
        for c in range(cols):
            if (r, c) in longest_path_set:
                output_grid.values[r][c] = 5  # Gray
            elif output_grid.values[r][c] == 0:
                output_grid.values[r][c] = 2  # Red

    return output_grid
