from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by introducing or modifying a secondary color (sky blue, magenta, or green)
    in a balanced pattern. The function follows these steps:
    1. Identifies the secondary color to use (8: sky blue, 6: magenta, or 3: green).
    2. Locates existing sections of the secondary color and 3x3 sections with blue squares.
    3. Applies transformations by shifting existing secondary color sections and introducing
       the secondary color in 2-3 additional 3x3 sections.
    4. Preserves the yellow grid structure and black squares.
    5. Ensures a visually balanced pattern with 6-8 total changes.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Identify the secondary color
    secondary_color = identify_secondary_color(output_grid)

    # Find existing secondary color sections and potential 3x3 sections for transformation
    existing_sections = find_existing_sections(output_grid, secondary_color)
    potential_sections = find_potential_sections(output_grid)

    # Apply transformations
    if existing_sections:
        shift_existing_sections(output_grid, existing_sections, secondary_color)
    
    transform_additional_sections(output_grid, potential_sections, secondary_color)

    return output_grid

def identify_secondary_color(grid: ColoredGrid) -> int:
    colors = grid.get_unique_colors()
    if 8 in colors:
        return 8  # sky blue
    elif 6 in colors:
        return 6  # magenta
    elif 3 in colors:
        return 3  # green
    else:
        return 8  # default to sky blue if no secondary color is present

def find_existing_sections(grid: ColoredGrid, color: int) -> List[Tuple[int, int, int, int]]:
    return [section for section in grid.detect_rectangles() if section[0] == color]

def find_potential_sections(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    return [(r, c) for r in range(0, rows, 4) for c in range(0, cols, 4) 
            if grid.get_cell(r, c) != 4 and any(grid.get_cell(r+i, c+j) == 1 
            for i in range(3) for j in range(3) if r+i < rows and c+j < cols)]

def shift_existing_sections(grid: ColoredGrid, sections: List[Tuple[int, int, int, int]], color: int):
    for _, top, left, bottom, right in sections:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid.get_cell(r, c) == color:
                    new_r, new_c = r + 1, c + 1
                    if new_r <= bottom and new_c <= right and grid.get_cell(new_r, new_c) != 4:
                        grid.set_cell(new_r, new_c, color)
                        grid.set_cell(r, c, 0 if grid.get_cell(r-1, c-1) == 4 else 1)

def transform_additional_sections(grid: ColoredGrid, sections: List[Tuple[int, int]], color: int):
    changes = 0
    for top, left in sections:
        if changes >= 3:
            break
        blue_cells = [(r, c) for r in range(top, top+3) for c in range(left, left+3) 
                      if grid.get_cell(r, c) == 1]
        if len(blue_cells) >= 2:
            for r, c in blue_cells[:2]:
                grid.set_cell(r, c, color)
            changes += 1
