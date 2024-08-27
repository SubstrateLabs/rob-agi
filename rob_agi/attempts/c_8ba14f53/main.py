from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple

def solve_8ba14f53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x9 input grid into a 3x3 output grid based on the following rules:
    1. Identify the two most prominent non-black colors in the input grid.
    2. Analyze each color's position, span, and density.
    3. Represent each color in the output grid, preserving vertical order.
    4. Place colors based on their horizontal position and span in the input.
    5. If a color spans more than half the input grid width, represent it with three cells horizontally in the output.
    6. Ensure both prominent colors are represented, even if minimally.
    7. Maintain vertical separation between colors if possible.
    8. Fill remaining cells with black (0).

    The transformation creates a representation of the two most prominent colors
    while maintaining their relative vertical positions and accurately reflecting
    their horizontal distribution in the input grid.
    """
    colors = analyze_grid(input_grid)
    sorted_colors = rank_colors(colors)
    output_values = create_output_grid(sorted_colors, input_grid.get_dimensions())
    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> Dict[int, Dict]:
    colors = {}
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color != 0:
                if color not in colors:
                    colors[color] = {'left': c, 'top': r, 'right': c, 'bottom': r, 'count': 0}
                info = colors[color]
                info['left'] = min(info['left'], c)
                info['top'] = min(info['top'], r)
                info['right'] = max(info['right'], c)
                info['bottom'] = max(info['bottom'], r)
                info['count'] += 1
    
    for color, info in colors.items():
        info['h_span'] = (info['right'] - info['left'] + 1) / cols
        info['v_span'] = (info['bottom'] - info['top'] + 1) / rows
        info['center'] = ((info['left'] + info['right']) / 2) / cols
    
    return colors

def rank_colors(colors: Dict[int, Dict]) -> List[Tuple[int, Dict]]:
    return sorted(colors.items(), key=lambda x: (-x[1]['count'], x[1]['top']))

def create_output_grid(sorted_colors: List[Tuple[int, Dict]], input_dims: Tuple[int, int]) -> List[List[int]]:
    output_values = [[0, 0, 0] for _ in range(3)]
    rows, cols = input_dims
    
    if len(sorted_colors) < 2:
        return output_values

    color1, info1 = sorted_colors[0]
    color2, info2 = sorted_colors[1]

    # Determine vertical positions
    pos1 = min(2, info1['top'] * 3 // rows)
    pos2 = min(2, info2['top'] * 3 // rows)

    # Place colors and extend horizontally if necessary
    output_values[pos1][0] = color1
    if info1['h_span'] > 0.5:
        output_values[pos1][1] = color1
        output_values[pos1][2] = color1

    if pos2 == pos1:
        pos2 = min(2, pos2 + 1)
    output_values[pos2][0] = color2
    if info2['h_span'] > 0.5:
        output_values[pos2][1] = color2
        output_values[pos2][2] = color2

    # Shift rows up if necessary
    while output_values[0] == [0, 0, 0] and any(row != [0, 0, 0] for row in output_values[1:]):
        output_values = output_values[1:] + [[0, 0, 0]]

    return output_values
