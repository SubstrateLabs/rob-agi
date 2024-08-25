from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_4e45f183(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying pattern-based transformations to 5x5 sections.
    
    The transformation involves:
    1. Creating frames in left and right sections with the less frequent color.
    2. Generating a symmetrical pattern in the middle section.
    3. Ensuring symmetry between top/bottom and left/right sections.
    4. Preserving color balance and adapting patterns to maintain symmetry.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    def get_section_colors(section: List[List[int]]) -> Tuple[int, int]:
        colors = [color for row in section for color in row if color != 0]
        color_counts = {color: colors.count(color) for color in set(colors)}
        sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], -x[0]))
        return sorted_colors[0][0], sorted_colors[1][0] if len(sorted_colors) > 1 else sorted_colors[0][0]

    def create_framed_section(frame_color: int, interior_color: int) -> List[List[int]]:
        section = [[interior_color] * 5 for _ in range(5)]
        for i in range(5):
            section[0][i] = section[4][i] = section[i][0] = section[i][4] = frame_color
        return section

    def create_middle_section(frame_color: int, interior_color: int) -> List[List[int]]:
        section = create_framed_section(frame_color, interior_color)
        section[2][2] = frame_color
        return section

    def transform_section(section: List[List[int]], position: str) -> List[List[int]]:
        dominant, secondary = get_section_colors(section)
        
        if position in ['left', 'right']:
            return create_framed_section(secondary, dominant)
        elif position == 'middle':
            return create_middle_section(secondary, dominant)
        else:
            return section

    rows, cols = input_grid.get_dimensions()
    output_values = [[0] * cols for _ in range(rows)]

    # Transform the top row of sections
    for j in range(3):
        section = input_grid.extract_subgrid(1, j*6+1, 5, 5).values
        position = ['left', 'middle', 'right'][j]
        transformed = transform_section(section, position)
        
        for r in range(5):
            for c in range(5):
                output_values[r+1][j*6+c+1] = transformed[r][c]

    # Mirror the top row to the bottom row
    for j in range(3):
        for r in range(5):
            for c in range(5):
                output_values[13+r][j*6+c+1] = output_values[5-r][j*6+c+1]

    # Transform the middle row of sections
    middle_section = input_grid.extract_subgrid(7, 7, 5, 5).values
    transformed_middle = transform_section(middle_section, 'middle')
    for r in range(5):
        for c in range(5):
            output_values[r+7][c+7] = transformed_middle[r][c]

    # Mirror the left section to the right section in the middle row
    for r in range(5):
        for c in range(5):
            output_values[r+7][13+c] = output_values[r+7][5-c]

    # Preserve black borders and separators
    for i in range(19):
        output_values[0][i] = output_values[18][i] = output_values[i][0] = output_values[i][18] = 0
        output_values[i][6] = output_values[i][12] = 0

    return ColoredGrid(values=output_values)
