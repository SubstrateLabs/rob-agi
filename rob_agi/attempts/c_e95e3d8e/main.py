from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying a repeating pattern
    in the non-black areas of the input grid, creating a template from this pattern,
    and using it to fill in the black areas while preserving the original non-black colors.
    
    1. Finds the smallest repeating pattern unit in non-black areas
    2. Creates a template by tiling this pattern unit
    3. Fills black areas with corresponding colors from the template
    4. Returns a new grid with the pattern completed
    """
    pattern_unit = find_pattern_unit(input_grid)
    template = create_pattern_template(pattern_unit, input_grid.num_rows, input_grid.num_cols)
    return fill_grid(input_grid, template)

def find_pattern_unit(input_grid: ColoredGrid) -> List[List[int]]:
    rows, cols = input_grid.get_dimensions()
    for unit_height in range(1, rows + 1):
        for unit_width in range(1, cols + 1):
            unit = extract_unit(input_grid, unit_height, unit_width)
            if is_valid_unit(input_grid, unit):
                return unit
    return input_grid.values  # Fallback to the entire grid if no smaller unit found

def extract_unit(grid: ColoredGrid, height: int, width: int) -> List[List[int]]:
    return [row[:width] for row in grid.values[:height]]

def is_valid_unit(grid: ColoredGrid, unit: List[List[int]]) -> bool:
    rows, cols = grid.get_dimensions()
    unit_height, unit_width = len(unit), len(unit[0])
    
    for i in range(rows):
        for j in range(cols):
            if grid.values[i][j] != 0:
                if grid.values[i][j] != unit[i % unit_height][j % unit_width]:
                    return False
    return True

def create_pattern_template(pattern_unit: List[List[int]], min_height: int, min_width: int) -> List[List[int]]:
    unit_height, unit_width = len(pattern_unit), len(pattern_unit[0])
    repeat_y = (min_height + unit_height - 1) // unit_height
    repeat_x = (min_width + unit_width - 1) // unit_width
    
    return [
        [pattern_unit[i % unit_height][j % unit_width] for j in range(repeat_x * unit_width)]
        for i in range(repeat_y * unit_height)
    ]

def fill_grid(input_grid: ColoredGrid, pattern_template: List[List[int]]) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    new_values = []
    
    for i in range(rows):
        new_row = []
        for j in range(cols):
            if input_grid.values[i][j] != 0:
                new_row.append(input_grid.values[i][j])
            else:
                new_row.append(pattern_template[i][j])
        new_values.append(new_row)
    
    return ColoredGrid(values=new_values)
