from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-8 pixels (inclusive) are changed to red or green.
    2. The target color (red or green) is determined by the global prevalence of these colors.
    3. Larger blue regions, single blue pixels, and other colors remain unchanged.
    4. All transformations are applied simultaneously.
    5. Gray (5) acts as a border and is not considered part of any region.
    6. Only orthogonally adjacent cells are considered part of the same region.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()
    regions_to_transform = []

    def flood_fill(start_row: int, start_col: int) -> List[Tuple[int, int]]:
        color = input_grid.values[start_row][start_col]
        stack = [(start_row, start_col)]
        region = []
        while stack:
            row, col = stack.pop()
            if (row, col) in visited or input_grid.values[row][col] != color:
                continue
            visited.add((row, col))
            region.append((row, col))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and input_grid.values[new_row][new_col] == color:
                    stack.append((new_row, new_col))
        return region

    def determine_target_color() -> int:
        red_count = sum(row.count(2) for row in input_grid.values)
        green_count = sum(row.count(3) for row in input_grid.values)
        return 2 if red_count >= green_count else 3

    target_color = determine_target_color()

    for row in range(rows):
        for col in range(cols):
            if input_grid.values[row][col] == 1 and (row, col) not in visited:
                region = flood_fill(row, col)
                if 2 <= len(region) <= 8:
                    regions_to_transform.append(region)

    for region in regions_to_transform:
        for r, c in region:
            output_grid.values[r][c] = target_color

    return output_grid
