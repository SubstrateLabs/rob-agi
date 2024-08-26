from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_42918530(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by making each 5x5 sub-grid rotationally symmetrical.
    The function preserves the general character and color count of each sub-grid
    while ensuring rotational symmetry. Black borders (0s) between sub-grids are preserved.
    
    1. Extracts 5x5 sub-grids from the input
    2. Analyzes each sub-grid based on its primary color
    3. Transforms sub-grids that require symmetry
    4. Reassembles the full grid with transformed sub-grids

    The transformation respects color-specific rules:
    - Red (2) and Green (3) are always made symmetrical
    - Blue (1) is never changed
    - Other colors are made symmetrical if they're not already
    """
    SUBGRID_SIZE = 5
    GRID_STEP = 6

    COLOR_RULES: Dict[int, str] = {
        1: "keep",  # Blue
        2: "symmetry",  # Red
        3: "symmetry",  # Green
    }

    def extract_subgrids(grid: List[List[int]]) -> List[List[List[int]]]:
        subgrids = []
        for i in range(0, len(grid), GRID_STEP):
            for j in range(0, len(grid[0]), GRID_STEP):
                subgrid = [row[j:j+SUBGRID_SIZE] for row in grid[i:i+SUBGRID_SIZE]]
                subgrids.append(subgrid)
        return subgrids

    def is_rotationally_symmetric(subgrid: List[List[int]]) -> bool:
        return all(subgrid[i][j] == subgrid[SUBGRID_SIZE-1-i][SUBGRID_SIZE-1-j] 
                   for i in range(SUBGRID_SIZE) for j in range(SUBGRID_SIZE))

    def count_color(subgrid: List[List[int]], color: int) -> int:
        return sum(row.count(color) for row in subgrid)

    def find_primary_color(subgrid: List[List[int]]) -> int:
        colors = set(cell for row in subgrid for cell in row if cell != 0)
        return max(colors, key=lambda c: count_color(subgrid, c))

    def generate_symmetric_pattern(color: int, count: int) -> List[List[int]]:
        pattern = [[0 for _ in range(SUBGRID_SIZE)] for _ in range(SUBGRID_SIZE)]
        positions = [(2,2), (1,1), (1,3), (3,1), (3,3), (0,0), (0,4), (4,0), (4,4),
                     (1,2), (2,1), (2,3), (3,2), (0,2), (2,0), (2,4), (4,2)]
        for i, j in positions[:count]:
            pattern[i][j] = pattern[SUBGRID_SIZE-1-i][SUBGRID_SIZE-1-j] = color
        return pattern

    def transform_subgrid(subgrid: List[List[int]]) -> List[List[int]]:
        color = find_primary_color(subgrid)
        rule = COLOR_RULES.get(color, "flexible")
        
        if rule == "keep" or (rule == "flexible" and is_rotationally_symmetric(subgrid)):
            return subgrid
        
        count = count_color(subgrid, color)
        return generate_symmetric_pattern(color, (count + 1) // 2)

    def reassemble_grid(subgrids: List[List[List[int]]], original_grid: List[List[int]]) -> List[List[int]]:
        new_grid = [row[:] for row in original_grid]
        subgrid_index = 0
        for i in range(0, len(original_grid), GRID_STEP):
            for j in range(0, len(original_grid[0]), GRID_STEP):
                for x in range(SUBGRID_SIZE):
                    for y in range(SUBGRID_SIZE):
                        new_grid[i+x][j+y] = subgrids[subgrid_index][x][y]
                subgrid_index += 1
        return new_grid

    subgrids = extract_subgrids(input_grid.values)
    transformed_subgrids = [transform_subgrid(subgrid) for subgrid in subgrids]
    new_grid_values = reassemble_grid(transformed_subgrids, input_grid.values)
    
    return ColoredGrid(values=new_grid_values)
