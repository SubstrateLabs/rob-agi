from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_42918530(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by making each 5x5 sub-grid rotationally symmetrical.
    The function preserves the general character and color count of each sub-grid
    while ensuring rotational symmetry. Black borders (0s) between sub-grids are preserved.
    
    1. Extracts 5x5 sub-grids from the input
    2. Checks each sub-grid for rotational symmetry
    3. If not symmetrical, generates a new symmetrical pattern
    4. Reassembles the full grid with transformed sub-grids
    """
    def extract_subgrids(grid: List[List[int]]) -> List[List[List[int]]]:
        subgrids = []
        for i in range(0, len(grid), 6):
            for j in range(0, len(grid[0]), 6):
                subgrid = [row[j:j+5] for row in grid[i:i+5]]
                subgrids.append(subgrid)
        return subgrids

    def is_rotationally_symmetric(subgrid: List[List[int]]) -> bool:
        return all(subgrid[i][j] == subgrid[4-i][4-j] for i in range(5) for j in range(5))

    def count_color(subgrid: List[List[int]], color: int) -> int:
        return sum(row.count(color) for row in subgrid)

    def find_primary_color(subgrid: List[List[int]]) -> int:
        colors = set(cell for row in subgrid for cell in row if cell != 0)
        return max(colors, key=lambda c: count_color(subgrid, c))

    def generate_symmetric_pattern(color: int, count: int) -> List[List[int]]:
        pattern = [[0 for _ in range(5)] for _ in range(5)]
        positions = [(2,2), (1,1), (1,3), (3,1), (3,3), (0,0), (0,4), (4,0), (4,4),
                     (1,2), (2,1), (2,3), (3,2), (0,2), (2,0), (2,4), (4,2)]
        for i, j in positions[:count]:
            pattern[i][j] = pattern[4-i][4-j] = color
        return pattern

    def transform_subgrid(subgrid: List[List[int]]) -> List[List[int]]:
        if is_rotationally_symmetric(subgrid):
            return subgrid
        color = find_primary_color(subgrid)
        count = count_color(subgrid, color)
        return generate_symmetric_pattern(color, (count + 1) // 2)

    def reassemble_grid(subgrids: List[List[List[int]]], original_grid: List[List[int]]) -> List[List[int]]:
        new_grid = [row[:] for row in original_grid]
        subgrid_index = 0
        for i in range(0, len(original_grid), 6):
            for j in range(0, len(original_grid[0]), 6):
                for x in range(5):
                    for y in range(5):
                        new_grid[i+x][j+y] = subgrids[subgrid_index][x][y]
                subgrid_index += 1
        return new_grid

    subgrids = extract_subgrids(input_grid.values)
    transformed_subgrids = [transform_subgrid(subgrid) for subgrid in subgrids]
    new_grid_values = reassemble_grid(transformed_subgrids, input_grid.values)
    
    return ColoredGrid(values=new_grid_values)
