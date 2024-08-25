from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_4e45f183(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying symmetrical pattern-based transformations to 5x5 sections.
    
    The transformation involves:
    1. Analyzing each 5x5 section to determine the two most frequent colors.
    2. Creating frames in left and right sections with the less frequent color.
    3. Generating a symmetrical pattern in the middle sections.
    4. Ensuring both horizontal and vertical symmetry across the entire grid.
    5. Preserving the original grid structure with black borders and separators.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with symmetrical patterns.
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
        section = [[interior_color] * 5 for _ in range(5)]
        section[0][0] = section[0][4] = section[4][0] = section[4][4] = frame_color
        section[2][2] = frame_color
        return section

    def transform_section(section: List[List[int]], position: str) -> List[List[int]]:
        primary, secondary = get_section_colors(section)
        if position in ['left', 'right']:
            return create_framed_section(secondary, primary)
        elif position == 'middle':
            return create_middle_section(secondary, primary)
        else:
            return section

    rows, cols = input_grid.get_dimensions()
    output_values = [[0] * cols for _ in range(rows)]

    # Transform and apply symmetry for each third of the grid
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
            
            # Apply vertical symmetry within each third
            if j == 0:  # Left section
                for r in range(5):
                    for c in range(5):
                        output_values[start_row+r][13+c] = transformed[r][4-c]

    # Apply horizontal symmetry
    for r in range(5):
        for c in range(17):
            output_values[13+r][c+1] = output_values[5-r][c+1]

    # Preserve black borders and separators
    for i in range(19):
        output_values[0][i] = output_values[18][i] = output_values[i][0] = output_values[i][18] = 0
        output_values[6][i] = output_values[12][i] = output_values[i][6] = output_values[i][12] = 0

    return ColoredGrid(values=output_values)
