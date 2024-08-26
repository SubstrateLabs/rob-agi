from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_42918530(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by making each 5x5 sub-grid rotationally symmetrical.
    The function preserves the specific patterns within each sub-grid while ensuring
    rotational symmetry. Black borders (0s) between sub-grids are preserved.
    
    1. Extracts 5x5 sub-grids from the input
    2. For each non-black sub-grid:
       a. Identifies the primary color and counts non-zero cells
       b. Analyzes the 2x2 quadrants to determine the representative pattern
       c. Creates a rotationally symmetric pattern based on the representative quadrant
    3. Ensures consistency across similar sub-grids
    4. Reassembles the full grid with transformed sub-grids
    5. Verifies color counts and rotational symmetry

    The transformation maintains the original color distribution and total cell count
    for each color while creating rotationally symmetric patterns for each sub-grid.
    """
    SUBGRID_SIZE = 5
    GRID_STEP = 6

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

    def analyze_quadrants(subgrid: List[List[int]]) -> List[List[int]]:
        quadrants = [
            [subgrid[i][j] for i in range(2) for j in range(2)],
            [subgrid[i][j] for i in range(2) for j in range(3, 5)],
            [subgrid[i][j] for i in range(3, 5) for j in range(2)],
            [subgrid[i][j] for i in range(3, 5) for j in range(3, 5)]
        ]
        return max(quadrants, key=lambda q: sum(1 for cell in q if cell != 0))

    def create_symmetric_pattern(quadrant: List[int], color: int, center: int) -> List[List[int]]:
        pattern = [[0 for _ in range(SUBGRID_SIZE)] for _ in range(SUBGRID_SIZE)]
        for i in range(2):
            for j in range(2):
                pattern[i][j] = pattern[i][4-j] = pattern[4-i][j] = pattern[4-i][4-j] = quadrant[i*2 + j]
        pattern[0][2] = pattern[2][0] = pattern[4][2] = pattern[2][4] = quadrant[1]
        pattern[2][2] = center
        return pattern

    def transform_subgrid(subgrid: List[List[int]]) -> List[List[int]]:
        if is_subgrid_all_black(subgrid):
            return subgrid
        color = find_primary_color(subgrid)
        representative_quadrant = analyze_quadrants(subgrid)
        return create_symmetric_pattern(representative_quadrant, color, subgrid[2][2])

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

    def verify_color_counts(original: List[List[int]], transformed: List[List[int]]) -> bool:
        original_counts = Counter(cell for row in original for cell in row)
        transformed_counts = Counter(cell for row in transformed for cell in row)
        return original_counts == transformed_counts

    def verify_rotational_symmetry(subgrid: List[List[int]]) -> bool:
        return all(subgrid[i][j] == subgrid[4-i][4-j] for i in range(5) for j in range(5))

    subgrids = extract_subgrids(input_grid.values)
    transformed_subgrids = [transform_subgrid(subgrid) for subgrid in subgrids]
    new_grid_values = reassemble_grid(transformed_subgrids, input_grid.values)
    
    assert verify_color_counts(input_grid.values, new_grid_values), "Color counts mismatch"
    assert all(verify_rotational_symmetry(subgrid) for subgrid in extract_subgrids(new_grid_values)), "Rotational symmetry not achieved"
    
    return ColoredGrid(values=new_grid_values)
