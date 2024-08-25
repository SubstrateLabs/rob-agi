from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_351d6448(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the pattern recognition challenge by identifying the object in each section,
    determining its growth or shift pattern, and extrapolating to the next state.
    The solution is then placed in a new 3x13 grid in the middle row.
    
    1. Identifies sections separated by gray lines
    2. Finds objects (non-black, non-gray) in each section
    3. Determines if objects are growing or shifting
    4. Extrapolates the next state of the object
    5. Places the extrapolated object in a new 3x13 grid
    """
    rows, cols = input_grid.get_dimensions()
    sections = get_sections(input_grid)
    objects = [find_object(input_grid, section) for section in sections]
    
    if is_growing_pattern(objects):
        next_object = extrapolate_growth(objects)
    else:
        next_object = extrapolate_shift(objects, cols)
    
    return create_output_grid(next_object, cols)

def get_sections(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Finds sections separated by gray lines (color 5)."""
    rows, _ = grid.get_dimensions()
    sections = []
    start = 0
    for r in range(rows):
        if all(cell == 5 for cell in grid.values[r]):
            if start != r:
                sections.append((start, r))
            start = r + 1
    if start != rows:
        sections.append((start, rows))
    return sections

def find_object(grid: ColoredGrid, section: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    """Finds non-black, non-gray cells in a section."""
    start, end = section
    return [(r, c, grid.values[r][c]) for r in range(start, end) 
            for c in range(len(grid.values[r])) 
            if grid.values[r][c] not in [0, 5]]

def is_growing_pattern(objects: List[List[Tuple[int, int, int]]]) -> bool:
    """Determines if the pattern is growing or shifting."""
    return len(set(len(obj) for obj in objects)) > 1

def extrapolate_growth(objects: List[List[Tuple[int, int, int]]]) -> List[Tuple[int, int, int]]:
    """Extrapolates the next state for a growing pattern."""
    growth_rate = len(objects[-1]) - len(objects[-2])
    next_size = len(objects[-1]) + growth_rate
    return [(0, c, objects[-1][0][2]) for c in range(next_size)]

def extrapolate_shift(objects: List[List[Tuple[int, int, int]]], cols: int) -> List[Tuple[int, int, int]]:
    """Extrapolates the next state for a shifting pattern."""
    shift = objects[-1][0][1] - objects[-2][0][1]
    next_start = (objects[-1][0][1] + shift) % cols
    return [(0, (c + next_start) % cols, cell[2]) for c, cell in enumerate(objects[-1])]

def create_output_grid(next_object: List[Tuple[int, int, int]], cols: int) -> ColoredGrid:
    """Creates a 3x13 output grid with the extrapolated object in the middle row."""
    output = [[0 for _ in range(cols)] for _ in range(3)]
    for _, c, color in next_object:
        if c < cols:
            output[1][c] = color
    return ColoredGrid(values=output)
