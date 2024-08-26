from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set

def solve_c663677b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the full repeating pattern
    and applying it to the entire grid, including black (0) areas.

    The solution follows these steps:
    1. Analyze the input grid to identify non-black areas and unique colors.
    2. Identify the pattern unit by gradually increasing window size.
    3. Check for symmetry and mirroring across quadrants.
    4. Simplify and validate the pattern.
    5. Reconstruct the full grid using the identified pattern.
    6. Perform final verification of the completed grid.

    Args:
        input_grid (ColoredGrid): The input grid with partial pattern and black areas.

    Returns:
        ColoredGrid: The solved grid with the full pattern applied to all cells.
    """
    # Step 1: Analyze the grid
    non_black_map, unique_colors = analyze_grid(input_grid)

    # Step 2 & 3: Identify pattern unit and check for symmetry
    pattern_unit = identify_pattern_unit(input_grid, non_black_map)

    # Step 4: Validate the pattern
    if not validate_pattern(input_grid, pattern_unit, non_black_map):
        raise ValueError("Unable to find a valid pattern")

    # Step 5: Reconstruct the full grid
    output_grid = reconstruct_grid(pattern_unit, input_grid.get_dimensions())

    # Step 6: Final verification
    if not verify_output(output_grid, unique_colors):
        raise ValueError("Output grid does not meet all criteria")

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

def validate_pattern(input_grid: ColoredGrid, pattern: List[List[int]], non_black_map: Dict[Tuple[int, int], int]) -> bool:
    rows, cols = input_grid.get_dimensions()
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in non_black_map:
                if input_grid.values[r][c] != pattern[r % pattern_rows][c % pattern_cols]:
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
def identify_pattern_unit(grid: ColoredGrid, non_black_map: Dict[Tuple[int, int], int]) -> List[List[int]]:
    rows, cols = grid.get_dimensions()
    for window_size in range(2, min(rows, cols) + 1):
        for r in range(rows - window_size + 1):
            for c in range(cols - window_size + 1):
                pattern = extract_pattern(grid, window_size, window_size, non_black_map, r, c)
                if validate_pattern(grid, pattern, non_black_map):
                    return pattern
    raise ValueError("No valid pattern unit found")

def extract_pattern(grid: ColoredGrid, height: int, width: int, non_black_map: Dict[Tuple[int, int], int], start_r: int = 0, start_c: int = 0) -> List[List[int]]:
    pattern = [[0 for _ in range(width)] for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if (start_r + r, start_c + c) in non_black_map:
                pattern[r][c] = non_black_map[(start_r + r, start_c + c)]
    return pattern

def reconstruct_grid(pattern: List[List[int]], dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])
    
    for r in range(rows):
        for c in range(cols):
            new_values[r][c] = pattern[r % pattern_rows][c % pattern_cols]
    
    return ColoredGrid(values=new_values)

def verify_output(grid: ColoredGrid, unique_colors: Set[int]) -> bool:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0 or grid.values[r][c] not in unique_colors:
                return False
    return True
