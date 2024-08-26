from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_81c0276b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting 2x2 colored squares from each section
    and arranging them in a compact output grid.

    The function works as follows:
    1. Identifies the frame color and sections in the input grid.
    2. For each section, extracts 2x2 colored squares, preserving their order.
    3. Creates an output grid where each row corresponds to a section.
    4. Fills the output grid with the colors found, maintaining their order and frequency.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    rows, cols = input_grid.get_dimensions()
    frame_color = identify_frame_color(input_grid)
    sections = get_sections(input_grid, frame_color)
    
    colors_per_section = []
    for section in sections:
        colors = extract_colors(section, frame_color)
        if colors:
            colors_per_section.append(colors)
    
    if not colors_per_section:
        return ColoredGrid(values=[[0]])
    
    max_width = max(sum(len(color_list) for color_list in section) for section in colors_per_section)
    output_values = []
    
    for section_colors in colors_per_section:
        row = []
        for color_list in section_colors:
            row.extend(color_list)
        row.extend([0] * (max_width - len(row)))
        output_values.append(row)
    
    return ColoredGrid(values=output_values)

def identify_frame_color(grid: ColoredGrid) -> int:
    """Identifies the frame color by checking the first row."""
    for color in grid.values[0]:
        if color != 0:
            return color
    return 0  # Default to black if no frame is found

def get_sections(grid: ColoredGrid, frame_color: int) -> List[ColoredGrid]:
    """Splits the grid into sections based on horizontal frame lines."""
    rows, cols = grid.get_dimensions()
    sections = []
    start_row = 0
    
    for r in range(rows):
        if all(cell == frame_color for cell in grid.values[r]):
            if r > start_row:
                sections.append(grid.extract_subgrid(start_row, 0, r - start_row, cols))
            start_row = r + 1
    
    if start_row < rows:
        sections.append(grid.extract_subgrid(start_row, 0, rows - start_row, cols))
    
    return sections

def extract_colors(section: ColoredGrid, frame_color: int) -> List[List[int]]:
    """Extracts 2x2 colored squares from a section, preserving their order."""
    rows, cols = section.get_dimensions()
    colors = []
    
    for c in range(0, cols - 1, 4):
        for r in range(0, rows - 1, 4):
            color = section.values[r][c]
            if color != 0 and color != frame_color:
                if all(section.values[r+i][c+j] == color for i in range(2) for j in range(2)):
                    colors.append([color, color])
    
    return colors
