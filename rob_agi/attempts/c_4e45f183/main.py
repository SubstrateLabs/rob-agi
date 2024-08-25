from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_4e45f183(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a pattern-based transformation to 5x5 sections.
    
    The transformation involves:
    1. Creating frames in left and right columns with the less frequent color.
    2. Inverting colors in the middle column.
    3. Ensuring symmetry between top/bottom and left/right sections.
    4. Preserving special patterns in the center section.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    def get_section_colors(section: List[List[int]]) -> Tuple[int, int]:
        colors = [color for row in section for color in row if color != 0]
        color_counts = {color: colors.count(color) for color in set(colors)}
        sorted_colors = sorted(color_counts.items(), key=lambda x: (-x[1], -x[0]))
        return sorted_colors[0][0], sorted_colors[1][0]

    def create_framed_section(pattern_color: int, background_color: int) -> List[List[int]]:
        section = [[background_color] * 5 for _ in range(5)]
        for i in range(5):
            section[0][i] = section[4][i] = section[i][0] = section[i][4] = pattern_color
        return section

    def invert_colors(section: List[List[int]]) -> List[List[int]]:
        background, pattern = get_section_colors(section)
        return [[pattern if cell == background else background if cell != 0 else 0 for cell in row] for row in section]

    def is_special_pattern(section: List[List[int]]) -> bool:
        return len(set(cell for row in section for cell in row if cell != 0)) > 2

    def transform_section(section: List[List[int]], position: str) -> List[List[int]]:
        if all(cell == section[0][0] for row in section for cell in row if cell != 0):
            return section
        
        pattern, background = get_section_colors(section)
        
        if position in ['left', 'right']:
            return create_framed_section(pattern, background)
        elif position == 'middle':
            if is_special_pattern(section):
                return invert_colors(section)
            else:
                return create_framed_section(background, pattern)
        else:
            return section

    rows, cols = input_grid.get_dimensions()
    output_values = [[0] * cols for _ in range(rows)]

    for i in range(3):
        for j in range(3):
            section = input_grid.extract_subgrid(i*6+1, j*6+1, 5, 5).values
            position = ['left', 'middle', 'right'][j]
            transformed = transform_section(section, position)
            
            if i == 2:  # Bottom row, mirror top row
                transformed = transform_section(input_grid.extract_subgrid(1, j*6+1, 5, 5).values, position)
            if j == 2:  # Right column, mirror left column
                transformed = transform_section(input_grid.extract_subgrid(i*6+1, 1, 5, 5).values, 'left')
            
            for r in range(5):
                for c in range(5):
                    output_values[i*6+r+1][j*6+c+1] = transformed[r][c]

    return ColoredGrid(values=output_values)
