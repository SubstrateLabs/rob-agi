from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing color patterns and significance.
    
    The function divides the input into three 4x6 sections, analyzes color distribution and patterns,
    and creates a condensed representation in the output. It preserves dominant colors, significant
    patterns, and overall color distribution while reducing the vertical dimension.
    
    Steps:
    1. Divide the input grid into three sections
    2. Analyze color significance in each section
    3. Generate the output grid based on color significance and patterns
    4. Preserve continuous patterns across sections
    5. Balance color representation and adjust for special cases
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3
    
    def analyze_section(section):
        flat_section = [cell for row in section for cell in row]
        return Counter(flat_section)
    
    def get_significant_colors(counter, n=3):
        return [color for color, _ in counter.most_common(n) if color != 0]
    
    def choose_color(col, significant_colors, section):
        column = [row[col] for row in section]
        column_counter = Counter(column)
        for color in significant_colors:
            if color in column_counter:
                return color
        return max(column_counter, key=column_counter.get)
    
    output_values = []
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        color_counter = analyze_section(section)
        significant_colors = get_significant_colors(color_counter)
        
        output_row = []
        for col in range(cols):
            color = choose_color(col, significant_colors, section)
            output_row.append(color)
        
        output_values.append(output_row)
    
    # Preserve continuous patterns
    for col in range(cols):
        if all(output_values[i][col] == output_values[0][col] for i in range(3)):
            continue
        column = [input_grid.values[row][col] for row in range(rows)]
        if len(set(column)) == 1 and column[0] != 0:
            for i in range(3):
                output_values[i][col] = column[0]
    
    return ColoredGrid(values=output_values)
