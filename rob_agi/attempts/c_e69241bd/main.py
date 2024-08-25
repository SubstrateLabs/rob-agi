from rob_agi.colored_grid import ColoredGrid
from collections import deque
from typing import List, Tuple

def solve_e69241bd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the e69241bd challenge by propagating non-zero, non-5 values within vertical sections
    separated by columns of 5s. The propagation respects section boundaries and gives precedence
    to smaller values during conflicts.

    1. Identify vertical sections separated by columns of 5s.
    2. For each section, apply a flood fill algorithm to propagate non-zero, non-5 values.
    3. During propagation, smaller values take precedence over larger ones.
    4. Generate the final output grid based on the propagation results.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed output grid.
    """
    def find_sections(grid: List[List[int]]) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        sections = []
        current_section = []
        for col in range(cols):
            if all(grid[row][col] == 5 for row in range(rows)):
                if current_section:
                    sections.append(current_section)
                    current_section = []
            else:
                current_section.append(col)
        if current_section:
            sections.append(current_section)
        return sections

    def flood_fill(grid: List[List[int]], start_row: int, start_col: int, value: int, section: List[int]):
        rows, cols = len(grid), len(grid[0])
        queue = deque([(start_row, start_col)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            row, col = queue.popleft()
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and new_col in section:
                    current_value = grid[new_row][new_col]
                    if current_value == 0 or (current_value != 5 and current_value > value):
                        grid[new_row][new_col] = value
                        queue.append((new_row, new_col))

    output_grid = input_grid.deep_copy().values
    rows, cols = len(output_grid), len(output_grid[0])
    sections = find_sections(output_grid)

    for section in sections:
        for row in range(rows):
            for col in section:
                value = output_grid[row][col]
                if value not in [0, 5]:
                    flood_fill(output_grid, row, col, value, section)

    return ColoredGrid(values=output_grid)
