from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c663677b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the full repeating pattern
    and applying it to the entire grid, including black (0) areas.

    The solution follows these steps:
    1. Analyze the input grid to find non-black cells and the largest contiguous non-black region.
    2. Determine the pattern size by checking for repeating units.
    3. Extract the base pattern from the input grid.
    4. Validate the pattern against non-black areas in the input grid.
    5. Generate the output grid by applying the validated pattern to all cells.

    Args:
        input_grid (ColoredGrid): The input grid with partial pattern and black areas.

    Returns:
        ColoredGrid: The solved grid with the full pattern applied to all cells.
    """
    # Step 1: Analyze the grid
    non_black_map, max_region = analyze_grid(input_grid)

    # Step 2: Determine pattern size
    pattern_size = find_pattern_size(input_grid, non_black_map, max_region)

    # Step 3: Extract base pattern
    base_pattern = extract_base_pattern(input_grid, pattern_size)

    # Step 4: Validate pattern
    while not validate_pattern(input_grid, base_pattern, pattern_size):
        pattern_size = (pattern_size[0] * 2, pattern_size[1] * 2)
        base_pattern = extract_base_pattern(input_grid, pattern_size)

    # Step 5: Generate output grid
    output_grid = generate_output_grid(base_pattern, pattern_size, input_grid.get_dimensions())

    return output_grid

def analyze_grid(grid: ColoredGrid) -> Tuple[Dict[Tuple[int, int], int], Tuple[int, int]]:
    non_black_map = {}
    max_region = (0, 0)
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                non_black_map[(r, c)] = grid.values[r][c]
                max_region = max(max_region, (r+1, c+1))
    
    return non_black_map, max_region

def find_pattern_size(grid: ColoredGrid, non_black_map: Dict[Tuple[int, int], int], max_region: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    for size in range(1, min(rows, cols) + 1):
        if all(grid.values[r][c] == grid.values[r % size][c % size] for (r, c) in non_black_map):
            return (size, size)
    return max_region

def extract_base_pattern(grid: ColoredGrid, pattern_size: Tuple[int, int]) -> List[List[int]]:
    return [[grid.values[r][c] for c in range(pattern_size[1])] for r in range(pattern_size[0])]

def validate_pattern(grid: ColoredGrid, pattern: List[List[int]], pattern_size: Tuple[int, int]) -> bool:
    rows, cols = grid.get_dimensions()
    return all(
        grid.values[r][c] == 0 or grid.values[r][c] == pattern[r % pattern_size[0]][c % pattern_size[1]]
        for r in range(rows) for c in range(cols)
    )

def generate_output_grid(pattern: List[List[int]], pattern_size: Tuple[int, int], grid_size: Tuple[int, int]) -> ColoredGrid:
    output_values = [
        [pattern[r % pattern_size[0]][c % pattern_size[1]] for c in range(grid_size[1])]
        for r in range(grid_size[0])
    ]
    return ColoredGrid(values=output_values)
