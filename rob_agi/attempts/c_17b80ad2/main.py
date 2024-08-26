from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors based on the following rules:
    1. Identify connected color regions, including adjacent columns.
    2. Sort regions by their top-most and then left-most positions.
    3. Extend colors from the bottom row upwards.
    4. Fill remaining space by extending color regions vertically and horizontally.
    5. Handle overlaps by giving precedence to higher or more left regions.
    6. Preserve original non-black cells in their positions.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    def get_region(start_row: int, start_col: int, color: int) -> List[Tuple[int, int]]:
        region = []
        queue = deque([(start_row, start_col)])
        visited = set()
        while queue:
            row, col = queue.popleft()
            if (row, col) in visited or input_grid.values[row][col] != color:
                continue
            visited.add((row, col))
            region.append((row, col))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < height and 0 <= new_col < width:
                    queue.append((new_row, new_col))
        return region

    # Identify and sort color regions
    regions = []
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] != 0 and (row, col) not in [cell for region in regions for cell in region]:
                color = input_grid.values[row][col]
                region = get_region(row, col, color)
                regions.append((color, region))
    
    regions.sort(key=lambda x: (min(cell[0] for cell in x[1]), min(cell[1] for cell in x[1])))

    # Process bottom row
    bottom_colors = [(col, input_grid.values[height-1][col]) for col in range(width) if input_grid.values[height-1][col] != 0]
    for col, color in bottom_colors:
        for row in range(height-1, -1, -1):
            if new_grid.values[row][col] == 0:
                new_grid.values[row][col] = color
            else:
                break

    # Fill remaining space
    for color, region in regions:
        top = min(cell[0] for cell in region)
        bottom = max(cell[0] for cell in region)
        left = min(cell[1] for cell in region)
        right = max(cell[1] for cell in region)

        # Extend upwards
        for col in range(left, right + 1):
            for row in range(top, -1, -1):
                if new_grid.values[row][col] == 0:
                    new_grid.values[row][col] = color
                else:
                    break

        # Extend downwards
        for col in range(left, right + 1):
            for row in range(bottom, height):
                if new_grid.values[row][col] == 0:
                    new_grid.values[row][col] = color
                else:
                    break

    # Preserve original non-black cells
    for row in range(height):
        for col in range(width):
            if input_grid.values[row][col] != 0:
                new_grid.values[row][col] = input_grid.values[row][col]

    return new_grid
