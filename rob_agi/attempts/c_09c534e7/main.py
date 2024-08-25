from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_09c534e7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying contiguous regions of the same color
    and updating their values based on a color sequence progression.
    
    The transformation follows these rules:
    1. Find all contiguous regions of the same color.
    2. Update the interior of each region with the next color in the sequence.
    3. Preserve the original color for border cells of each region.
    4. Expand higher-value isolated cells to 2x2 areas where possible.
    5. Ensure no value in the output is less than its corresponding value in the input.
    6. Maintain the overall shape and connectivity of regions.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    color_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    processed = set()

    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if (r, c) not in processed and input_grid.values[r][c] != 0:
                region = find_region(input_grid, r, c)
                process_region(output_grid, region, color_sequence)
                processed.update(region)

    expand_higher_value_cells(output_grid)
    ensure_no_decrease(input_grid, output_grid)
    reconnect_borders(output_grid)
    return output_grid

def find_region(grid: ColoredGrid, r: int, c: int) -> Set[Tuple[int, int]]:
    color = grid.values[r][c]
    region = set()
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if (r, c) not in region and 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == color:
            region.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))
    return region

def process_region(grid: ColoredGrid, region: Set[Tuple[int, int]], color_sequence: List[int]):
    color = grid.values[list(region)[0][0]][list(region)[0][1]]
    next_color = color_sequence[(color_sequence.index(color) + 1) % len(color_sequence)]
    border = find_border(grid, region)
    for r, c in region - border:
        grid.values[r][c] = next_color
    for r, c in border:
        grid.values[r][c] = color

def find_border(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    border = set()
    for r, c in region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region or nr < 0 or nr >= grid.num_rows or nc < 0 or nc >= grid.num_cols:
                border.add((r, c))
                break
    return border

def expand_higher_value_cells(grid: ColoredGrid):
    for r in range(grid.num_rows - 1):
        for c in range(grid.num_cols - 1):
            max_value = max(grid.values[r][c], grid.values[r][c+1], grid.values[r+1][c], grid.values[r+1][c+1])
            if max_value > 1:
                grid.values[r][c] = max_value
                grid.values[r][c+1] = max_value
                grid.values[r+1][c] = max_value
                grid.values[r+1][c+1] = max_value

def ensure_no_decrease(input_grid: ColoredGrid, output_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            output_grid.values[r][c] = max(output_grid.values[r][c], input_grid.values[r][c])

def reconnect_borders(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] > 1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] == 0:
                        grid.values[nr][nc] = grid.values[r][c] - 1
