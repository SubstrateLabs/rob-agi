from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple

def solve_8ba14f53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x9 input grid into a 3x3 output grid based on the following rules:
    1. Identify the two most prominent non-black colors in the input grid.
    2. Analyze each color's shape characteristics (extent, position, density).
    3. Calculate the prominence ratio of the two colors.
    4. Allocate cells in the output grid based on color prominence and shape.
    5. Represent each color in its allocated cell(s) based on its characteristics.
    6. Fill any remaining cells with black (0).

    The transformation preserves the relative positions and basic shape features
    of the two most prominent colors while simplifying the overall representation.
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
    total_count = info1['count'] + info2['count']
    ratio1 = info1['count'] / total_count
    ratio2 = info2['count'] / total_count

    # Allocate cells for the first color
    if ratio1 > 0.7:
        output_values[0] = [color1, color1, color1]
        output_values[1] = [color1, color1, color1]
    elif ratio1 > 0.5:
        output_values[0] = [color1, color1, color1]
        if info1['h_span'] > 0.7:
            output_values[1] = [color1, color1, 0]
        elif info1['center'] < 1/3:
            output_values[1] = [color1, 0, 0]
        elif info1['center'] > 2/3:
            output_values[1] = [0, 0, color1]
        else:
            output_values[1] = [0, color1, 0]
    else:
        if info1['h_span'] > 0.7:
            output_values[0] = [color1, color1, color1]
        elif info1['center'] < 1/3:
            output_values[0] = [color1, color1, 0]
        elif info1['center'] > 2/3:
            output_values[0] = [0, color1, color1]
        else:
            output_values[0] = [color1, color1, 0]

    # Allocate cells for the second color
    if ratio2 > 0.3:
        if all(cell == 0 for cell in output_values[1]):
            output_values[1] = [color2, color2, color2]
        elif all(cell == 0 for cell in output_values[2]):
            output_values[2] = [color2, color2, color2]
    else:
        for row in range(3):
            if all(cell == 0 for cell in output_values[row]):
                if info2['h_span'] > 0.7:
                    output_values[row] = [color2, color2, color2]
                elif info2['center'] < 1/3:
                    output_values[row] = [color2, 0, 0]
                elif info2['center'] > 2/3:
                    output_values[row] = [0, 0, color2]
                else:
                    output_values[row] = [0, color2, 0]
                break

    # Ensure both colors are represented
    if color1 not in [cell for row in output_values for cell in row]:
        for row in range(3):
            if 0 in output_values[row]:
                output_values[row][output_values[row].index(0)] = color1
                break
    if color2 not in [cell for row in output_values for cell in row]:
        for row in range(3):
            if 0 in output_values[row]:
                output_values[row][output_values[row].index(0)] = color2
                break

    # Shift representation upwards if needed
    while all(cell == 0 for cell in output_values[0]):
        output_values = output_values[1:] + [[0, 0, 0]]

    return output_values
