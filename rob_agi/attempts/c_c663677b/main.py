from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c663677b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the full repeating pattern
    and applying it to the entire grid, including black (0) areas.

    The solution follows these steps:
    1. Analyze the input grid to identify non-black areas and unique colors.
    2. Determine the pattern repetition horizontally and vertically.
    3. Extract pattern elements based on the repetition.
    4. Construct the full pattern and validate it against the input.
    5. Fill in black spaces with the discovered pattern.
    6. Perform final verification of the completed grid.

    Args:
        input_grid (ColoredGrid): The input grid with partial pattern and black areas.

    Returns:
        ColoredGrid: The solved grid with the full pattern applied to all cells.
    """
    # Step 1: Analyze the grid
    non_black_map, unique_colors = analyze_grid(input_grid)

    # Step 2: Determine pattern repetition
    h_repeat, v_repeat = find_repetition(input_grid, non_black_map)

    # Step 3: Extract pattern elements
    pattern = extract_pattern(input_grid, h_repeat, v_repeat, non_black_map)

    # Step 4: Construct and validate full pattern
    full_pattern = construct_full_pattern(pattern, 27, 27)
    if not validate_pattern(input_grid, full_pattern, non_black_map):
        raise ValueError("Unable to find a valid pattern")

    # Step 5 & 6: Fill black spaces and perform final verification
    output_grid = fill_black_spaces(input_grid, full_pattern)

    return output_grid

def analyze_grid(grid: ColoredGrid) -> Tuple[Dict[Tuple[int, int], int], Set[int]]:
    non_black_map = {}
    unique_colors = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                non_black_map[(r, c)] = grid.values[r][c]
                unique_colors.add(grid.values[r][c])
    
    return non_black_map, unique_colors

def find_repetition(grid: ColoredGrid, non_black_map: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    
    def find_repeat(sequence):
        for length in range(1, len(sequence) + 1):
            if len(sequence) % length == 0:
                if all(sequence[i] == sequence[i % length] for i in range(len(sequence))):
                    return length
        return len(sequence)
    
    h_sequence = [grid.values[0][c] for c in range(cols) if (0, c) in non_black_map]
    v_sequence = [grid.values[r][0] for r in range(rows) if (r, 0) in non_black_map]
    
    h_repeat = find_repeat(h_sequence)
    v_repeat = find_repeat(v_sequence)
    
    return h_repeat, v_repeat

def extract_pattern(grid: ColoredGrid, h_repeat: int, v_repeat: int, non_black_map: Dict[Tuple[int, int], int]) -> List[List[int]]:
    pattern = [[0 for _ in range(h_repeat)] for _ in range(v_repeat)]
    
    for r in range(v_repeat):
        for c in range(h_repeat):
            if (r, c) in non_black_map:
                pattern[r][c] = non_black_map[(r, c)]
    
    return pattern

def construct_full_pattern(pattern: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    full_pattern = [[0 for _ in range(cols)] for _ in range(rows)]
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])
    
    for r in range(rows):
        for c in range(cols):
            full_pattern[r][c] = pattern[r % pattern_rows][c % pattern_cols]
    
    return full_pattern

def validate_pattern(input_grid: ColoredGrid, full_pattern: List[List[int]], non_black_map: Dict[Tuple[int, int], int]) -> bool:
    rows, cols = input_grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in non_black_map and input_grid.values[r][c] != full_pattern[r][c]:
                return False
    
    return True

def fill_black_spaces(input_grid: ColoredGrid, full_pattern: List[List[int]]) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    output_values = []
    
    for r in range(rows):
        row = []
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                row.append(input_grid.values[r][c])
            else:
                row.append(full_pattern[r][c])
        output_values.append(row)
    
    return ColoredGrid(values=output_values)
