from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_42918530(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by making each 5x5 sub-grid rotationally symmetrical.
    The function preserves the general character and color count of each sub-grid
    while ensuring rotational symmetry. Black borders (0s) between sub-grids are preserved.
    
    1. Analyzes the entire grid to determine standard patterns for each color and cell count
    2. Extracts 5x5 sub-grids from the input
    3. For each non-black sub-grid:
       a. Identifies the primary color and counts non-zero cells
       b. Applies a symmetric pattern based on the color and cell count
    4. Ensures consistency across similar sub-grids
    5. Reassembles the full grid with transformed sub-grids

    The transformation creates consistent symmetric patterns for each color
    across all transformed sub-grids while maintaining the original color distribution
    and total cell count for each color.
    """
    SUBGRID_SIZE = 5
    GRID_STEP = 6

    def analyze_grid(grid: List[List[int]]) -> Dict[Tuple[int, int], List[List[int]]]:
        color_patterns = {}
        subgrids = extract_subgrids(grid)
        
        for subgrid in subgrids:
            if not is_subgrid_all_black(subgrid):
                color = find_primary_color(subgrid)
                count = count_non_black_cells(subgrid)
                key = (color, count)
                if key not in color_patterns:
                    color_patterns[key] = generate_symmetric_pattern(color, count)
        
        return color_patterns

    def extract_subgrids(grid: List[List[int]]) -> List[List[List[int]]]:
        return [
            [row[j:j+SUBGRID_SIZE] for row in grid[i:i+SUBGRID_SIZE]]
            for i in range(0, len(grid), GRID_STEP)
            for j in range(0, len(grid[0]), GRID_STEP)
        ]

    def is_subgrid_all_black(subgrid: List[List[int]]) -> bool:
        return all(cell == 0 for row in subgrid for cell in row)

    def count_non_black_cells(subgrid: List[List[int]]) -> int:
        return sum(cell != 0 for row in subgrid for cell in row)

    def find_primary_color(subgrid: List[List[int]]) -> int:
        colors = [cell for row in subgrid for cell in row if cell != 0]
        return max(set(colors), key=colors.count) if colors else 0

    def generate_symmetric_pattern(color: int, count: int) -> List[List[int]]:
        pattern = [[0 for _ in range(SUBGRID_SIZE)] for _ in range(SUBGRID_SIZE)]
        positions = [(2,2), (0,0), (0,4), (4,0), (4,4), (0,2), (2,0), (2,4), (4,2),
                     (1,1), (1,3), (3,1), (3,3), (1,2), (2,1), (2,3), (3,2)]
        for i, j in positions[:count]:
            pattern[i][j] = color
        return pattern

    def transform_subgrid(subgrid: List[List[int]], color_patterns: Dict[Tuple[int, int], List[List[int]]]) -> List[List[int]]:
        if is_subgrid_all_black(subgrid):
            return subgrid
        color = find_primary_color(subgrid)
        count = count_non_black_cells(subgrid)
        return color_patterns.get((color, count), subgrid)

    def reassemble_grid(subgrids: List[List[List[int]]], original_grid: List[List[int]]) -> List[List[int]]:
        new_grid = [row[:] for row in original_grid]
        subgrid_index = 0
        for i in range(0, len(original_grid), GRID_STEP):
            for j in range(0, len(original_grid[0]), GRID_STEP):
                if subgrid_index < len(subgrids):
                    for x in range(SUBGRID_SIZE):
                        for y in range(SUBGRID_SIZE):
                            new_grid[i+x][j+y] = subgrids[subgrid_index][x][y]
                    subgrid_index += 1
        return new_grid

    color_patterns = analyze_grid(input_grid.values)
    subgrids = extract_subgrids(input_grid.values)
    transformed_subgrids = [transform_subgrid(subgrid, color_patterns) for subgrid in subgrids]
    new_grid_values = reassemble_grid(transformed_subgrids, input_grid.values)
    
    return ColoredGrid(values=new_grid_values)
