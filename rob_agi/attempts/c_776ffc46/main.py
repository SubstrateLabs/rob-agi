from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Connected blue regions are changed to red or green if they are adjacent to that color.
    2. The target color (red or green) is determined by the presence and adjacency of that color to the blue region.
    3. If a blue region is not adjacent to red or green, it remains blue.
    4. Gray areas and other colors remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_connected_region(row: int, col: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        region = []
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] == color and (r, c) not in visited:
                region.append((r, c))
                visited.add((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return region

    def is_adjacent_to_color(region: List[Tuple[int, int]], color: int) -> bool:
        for r, c in region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] == color:
                    return True
        return False

    visited = set()
    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1 and (row, col) not in visited:
                region = find_connected_region(row, col, 1, visited)
                if is_adjacent_to_color(region, 2):
                    target_color = 2
                elif is_adjacent_to_color(region, 3):
                    target_color = 3
                else:
                    continue  # Keep the region blue if not adjacent to red or green
                for r, c in region:
                    output_grid.values[r][c] = target_color

    return output_grid
