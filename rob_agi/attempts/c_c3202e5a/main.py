from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_c3202e5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by identifying a focus section and color, then creating a new grid
    based on the pattern of the focus color in that section.

    1. Analyzes the input grid to find dividing lines and section size.
    2. Identifies the focus section with the highest concentration of a single color.
    3. Determines the focus color within that section.
    4. Creates a new grid:
       - If input section is 4x4, expands to 5x5
       - If input section is 5x5, contracts to 3x3
    5. Transfers the pattern of the focus color to the new grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    dividing_color = find_dividing_lines(input_grid)
    section_size = get_section_size(input_grid, dividing_color)
    focus_section = find_focus_section(input_grid, dividing_color, section_size)
    focus_color = get_focus_color(focus_section)

    if section_size == 4:
        output_size = 5
        output_values = expand_pattern(focus_section, focus_color)
    elif section_size == 5:
        output_size = 3
        output_values = contract_pattern(focus_section, focus_color)
    else:
        raise ValueError(f"Unexpected section size: {section_size}")

    return ColoredGrid(values=output_values)

def find_dividing_lines(grid: ColoredGrid) -> int:
    """Identifies the color of the dividing lines."""
    for color in range(1, 10):  # Skip black (0)
        if all(cell == color for cell in grid.values[3]):  # Check 4th row
            return color
    raise ValueError("No dividing lines found")

def get_section_size(grid: ColoredGrid, dividing_color: int) -> int:
    """Calculates the size of individual sections."""
    for i in range(1, len(grid.values)):
        if grid.values[i][0] == dividing_color:
            return i
    raise ValueError("Could not determine section size")

def find_focus_section(grid: ColoredGrid, dividing_color: int, section_size: int) -> List[List[int]]:
    """Locates the section with the highest color concentration."""
    max_concentration = 0
    focus_section = None
    for i in range(0, len(grid.values), section_size + 1):
        for j in range(0, len(grid.values[0]), section_size + 1):
            section = [row[j:j+section_size] for row in grid.values[i:i+section_size]]
            concentration = max(count_color(section, color) for color in range(10) if color != dividing_color)
            if concentration > max_concentration:
                max_concentration = concentration
                focus_section = section
    if focus_section is None:
        raise ValueError("No focus section found")
    return focus_section

def get_focus_color(section: List[List[int]]) -> int:
    """Determines the most frequent non-zero color in a section."""
    color_counts = {color: count_color(section, color) for color in range(1, 10)}
    return max(color_counts, key=color_counts.get)

def count_color(section: List[List[int]], color: int) -> int:
    """Counts occurrences of a color in a section."""
    return sum(row.count(color) for row in section)

def expand_pattern(section: List[List[int]], focus_color: int) -> List[List[int]]:
    """Expands a 4x4 pattern to 5x5."""
    output = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(4):
        for j in range(4):
            output[i+1][j+1] = focus_color if section[i][j] == focus_color else 0
    # Extend pattern to extra row and column
    for i in range(1, 5):
        output[0][i] = output[1][i]
        output[i][0] = output[i][1]
    return output

def contract_pattern(section: List[List[int]], focus_color: int) -> List[List[int]]:
    """Contracts a 5x5 pattern to 3x3."""
    return [[focus_color if section[i+1][j+1] == focus_color else 0 for j in range(3)] for i in range(3)]
