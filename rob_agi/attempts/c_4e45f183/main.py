from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List
from collections import Counter

def solve_4e45f183(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying symmetrical pattern-based transformations to 5x5 sections.
    
    The transformation involves:
    1. Analyzing each 5x5 section to determine the two most frequent colors.
    2. Creating frame-like patterns in left and right sections with the less frequent color.
    3. Generating a symmetrical pattern in the middle sections.
    4. Ensuring both horizontal and vertical symmetry across the entire grid.
    5. Preserving the original grid structure with black borders and separators.
    6. Applying modified horizontal symmetry between top and bottom thirds.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with symmetrical patterns.
    """
    def get_section_colors(section: List[List[int]]) -> Tuple[int, int]:
        colors = [color for row in section for color in row if color != 0]
        color_counts = Counter(colors)
        sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], -x[0]))
        return sorted_colors[0][0], sorted_colors[1][0] if len(sorted_colors) > 1 else sorted_colors[0][0]

    def create_left_section(primary: int, secondary: int) -> List[List[int]]:
        section = [[primary] * 5 for _ in range(5)]
        for i in range(5):
            section[0][i] = section[i][0] = secondary
        section[4][4] = secondary
        return section

    def create_middle_section(primary: int, secondary: int) -> List[List[int]]:
        section = [[primary] * 5 for _ in range(5)]
        for i in range(5):
            section[0][i] = section[4][i] = secondary
        section[1][1] = section[1][3] = section[3][1] = section[3][3] = section[2][2] = secondary
        return section

    def create_right_section(primary: int, secondary: int) -> List[List[int]]:
        section = [[primary] * 5 for _ in range(5)]
        for i in range(5):
            section[0][i] = section[i][4] = secondary
        section[4][0] = secondary
        return section

    def transform_section(section: List[List[int]], position: str) -> List[List[int]]:
        primary, secondary = get_section_colors(section)
        if position == 'left':
            return create_left_section(primary, secondary)
        elif position == 'middle':
            return create_middle_section(primary, secondary)
        elif position == 'right':
            return create_right_section(primary, secondary)
        else:
            return section

    rows, cols = input_grid.get_dimensions()
    output_values = [[0] * cols for _ in range(rows)]

    # Transform each third of the grid
    for i in range(3):
        start_row = i * 6 + 1
        for j in range(3):
            section = input_grid.extract_subgrid(start_row, j*6+1, 5, 5).values
            position = ['left', 'middle', 'right'][j]
            transformed = transform_section(section, position)
            
            # Apply the transformed section
            for r in range(5):
                for c in range(5):
                    output_values[start_row+r][j*6+c+1] = transformed[r][c]

    # Apply vertical symmetry
    for i in range(3):
        start_row = i * 6 + 1
        for r in range(5):
            for c in range(5):
                output_values[start_row+r][13+c] = output_values[start_row+r][5-c]

    # Apply modified horizontal symmetry
    for r in range(5):
        for c in range(17):
            if c < 6:
                output_values[13+r][c+1] = output_values[5-r][c+1]
            elif 6 < c < 12:
                output_values[13+r][c+1] = output_values[7+r][c+1]
            else:
                output_values[13+r][c+1] = output_values[5-r][18-c]

    # Preserve black borders and separators
    for i in range(19):
        output_values[0][i] = output_values[18][i] = output_values[i][0] = output_values[i][18] = 0
        output_values[6][i] = output_values[12][i] = output_values[i][6] = output_values[i][12] = 0

    return ColoredGrid(values=output_values)
