from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c663677b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying the full repeating pattern
    and applying it to the entire grid, including black (0) areas.

    The solution follows these steps:
    1. Analyze the input grid to identify non-black areas.
    2. Identify 3x9 vertical strips from the non-black areas.
    3. Construct a 9x9 unit from the identified strips.
    4. Validate the 9x9 unit against the input grid.
    5. Generate the complete 27x27 pattern using the validated 9x9 unit.
    6. Fill in black spaces while preserving non-black areas from the input.

    Args:
        input_grid (ColoredGrid): The input grid with partial pattern and black areas.

    Returns:
        ColoredGrid: The solved grid with the full pattern applied to all cells.
    """
    # Step 1: Analyze the grid
    non_black_map = analyze_grid(input_grid)

    # Step 2: Identify 3x9 vertical strips
    strips = identify_strips(input_grid, non_black_map)

    # Step 3: Construct 9x9 unit
    unit = construct_unit(strips)

    # Step 4: Validate 9x9 unit
    if not validate_unit(input_grid, unit):
        raise ValueError("Unable to find a valid 9x9 unit")

    # Step 5 & 6: Generate output grid and fill black spaces
    output_grid = generate_output_grid(input_grid, unit)

    return output_grid

def analyze_grid(grid: ColoredGrid) -> Dict[Tuple[int, int], int]:
    non_black_map = {}
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                non_black_map[(r, c)] = grid.values[r][c]
    
    return non_black_map

def identify_strips(grid: ColoredGrid, non_black_map: Dict[Tuple[int, int], int]) -> List[List[List[int]]]:
    strips = []
    rows, cols = grid.get_dimensions()
    
    for c in range(0, cols, 3):
        strip = []
        for r in range(rows):
            row = [grid.values[r][c+i] if (r, c+i) in non_black_map else 0 for i in range(3)]
            strip.append(row)
        if any(any(cell != 0 for cell in row) for row in strip):
            strips.append(strip)
    
    return strips

def construct_unit(strips: List[List[List[int]]]) -> List[List[int]]:
    unit = []
    for i in range(0, 27, 9):
        for j in range(9):
            row = []
            for strip in strips:
                row.extend(strip[i+j])
            unit.append(row)
    return unit

def validate_unit(grid: ColoredGrid, unit: List[List[int]]) -> bool:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0 and grid.values[r][c] != unit[r % 9][c % 9]:
                return False
    return True

def generate_output_grid(input_grid: ColoredGrid, unit: List[List[int]]) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    output_values = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                row.append(input_grid.values[r][c])
            else:
                row.append(unit[r % 9][c % 9])
        output_values.append(row)
    return ColoredGrid(values=output_values)
