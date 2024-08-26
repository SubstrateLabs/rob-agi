from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating a template from non-black areas,
    determining the pattern dimensions, extending the template if necessary,
    and filling in the black areas with the corresponding pattern values.
    
    1. Creates a template from non-black areas of the input grid
    2. Determines the pattern dimensions based on non-black cells
    3. Extends the template to cover the entire grid if necessary
    4. Fills black areas with corresponding colors from the extended template
    5. Returns a new grid with the pattern completed
    """
    template = create_template(input_grid)
    pattern_width, pattern_height = get_pattern_dimensions(input_grid)
    extended_template = extend_template(template, pattern_width, pattern_height, input_grid.num_rows, input_grid.num_cols)
    return fill_grid(input_grid, extended_template)

def create_template(input_grid: ColoredGrid) -> List[List[int]]:
    return [[cell if cell != 0 else -1 for cell in row] for row in input_grid.values]

def get_pattern_dimensions(input_grid: ColoredGrid) -> Tuple[int, int]:
    rows, cols = input_grid.get_dimensions()
    pattern_width = max(j for i in range(rows) for j in range(cols) if input_grid.values[i][j] != 0) + 1
    pattern_height = max(i for i in range(rows) for j in range(cols) if input_grid.values[i][j] != 0) + 1
    return pattern_width, pattern_height

def extend_template(template: List[List[int]], pattern_width: int, pattern_height: int, rows: int, cols: int) -> List[List[int]]:
    extended = [[-1 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if template[i % pattern_height][j % pattern_width] != -1:
                extended[i][j] = template[i % pattern_height][j % pattern_width]
    return extended

def fill_grid(input_grid: ColoredGrid, extended_template: List[List[int]]) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    new_values = []
    for i in range(rows):
        new_row = []
        for j in range(cols):
            if input_grid.values[i][j] != 0:
                new_row.append(input_grid.values[i][j])
            else:
                new_row.append(extended_template[i][j])
        new_values.append(new_row)
    return ColoredGrid(values=new_values)
