from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing and condensing color patterns.
    
    The function divides the input into three 4x6 sections and creates a condensed representation in the output.
    Each row of the output represents its corresponding section of the input, while also
    considering influences from other sections for continuity and pattern preservation.
    
    Key steps:
    1. Analyze each third of the input grid for color frequency, patterns, and relationships
    2. Generate each output row based on its corresponding input section and inter-section relationships
    3. Incorporate vertical, horizontal, and diagonal continuity from the input
    4. Preserve unique patterns, symmetry, and color distributions from the input
    5. Balance color representation across the entire output
    6. Apply special rules for corners and edges
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3

    def analyze_section(section: List[List[int]]) -> Counter:
        return Counter(cell for row in section for cell in row if cell != 0)

    def get_vertical_pattern(grid: List[List[int]], col: int) -> List[int]:
        return [row[col] for row in grid if row[col] != 0]

    def get_diagonal_pattern(grid: List[List[int]], row: int, col: int, direction: int) -> List[int]:
        pattern = []
        r, c = row, col
        while 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            if grid[r][c] != 0:
                pattern.append(grid[r][c])
            r += 1
            c += direction
        return pattern

    def get_horizontal_pattern(grid: List[List[int]], row: int) -> List[int]:
        return [cell for cell in grid[row] if cell != 0]

    output_values = []
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_counter = analyze_section(section)
        
        output_row = []
        for col in range(cols):
            vertical_pattern = get_vertical_pattern(input_grid.values, col)
            diagonal_pattern_right = get_diagonal_pattern(input_grid.values, i*section_height, col, 1)
            diagonal_pattern_left = get_diagonal_pattern(input_grid.values, i*section_height, col, -1)
            horizontal_pattern = get_horizontal_pattern(input_grid.values, i*section_height)
            
            color_scores = {}
            for color in set(section_counter.keys()) | set(vertical_pattern) | set(diagonal_pattern_right) | set(diagonal_pattern_left) | set(horizontal_pattern):
                score = section_counter[color] * 3  # Primary weight to section colors
                score += vertical_pattern.count(color) * 2  # Secondary weight to vertical continuity
                score += diagonal_pattern_right.count(color) + diagonal_pattern_left.count(color)  # Weight to diagonal patterns
                score += horizontal_pattern.count(color)  # Weight to horizontal patterns
                color_scores[color] = score
            
            chosen_color = max(color_scores, key=color_scores.get) if color_scores else 0
            output_row.append(chosen_color)
        
        output_values.append(output_row)

    # Preserve unique patterns and apply symmetry
    for col in range(cols):
        column = [row[col] for row in input_grid.values]
        if len(set(column)) == 1 and column[0] != 0:
            for i in range(3):
                output_values[i][col] = column[0]
        elif column[:6] == column[6:]:  # Vertical symmetry
            output_values[1][col] = output_values[0][col]
            output_values[2][col] = output_values[0][col]

    # Apply horizontal symmetry if detected
    if input_grid.values[:4] == input_grid.values[8:]:
        output_values[2] = output_values[0]

    # Special rules for corners and edges
    for i in range(3):
        output_values[i][0] = input_grid.values[i*section_height][0]  # Left edge
        output_values[i][-1] = input_grid.values[i*section_height + section_height - 1][-1]  # Right edge

    output_values[0][0] = input_grid.values[0][0]  # Top-left corner
    output_values[0][-1] = input_grid.values[0][-1]  # Top-right corner
    output_values[-1][0] = input_grid.values[-1][0]  # Bottom-left corner
    output_values[-1][-1] = input_grid.values[-1][-1]  # Bottom-right corner

    # Balance color representation
    overall_counter = analyze_section(input_grid.values)
    for i, row in enumerate(output_values):
        row_counter = Counter(row)
        for color in overall_counter:
            if color not in row_counter and color != 0:
                least_common = min(row, key=row.count)
                row[row.index(least_common)] = color

    return ColoredGrid(values=output_values)
