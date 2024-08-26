from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple

def solve_8ba14f53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x9 input grid into a 3x3 output grid based on the following rules:
    1. Identify the three most prominent non-black colors in the input grid.
    2. Analyze each color's shape characteristics (extent, edge touching, density).
    3. Rank colors based on prominence and vertical position.
    4. Allocate cells in the output grid based on color ranking and input shape.
    5. Represent each color in its allocated cell(s) based on its characteristics.
    6. Fill any remaining cells with black (0).

    The transformation preserves the relative positions and basic shape features
    of the most prominent colors while simplifying the overall representation.
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
    
    for i, (color, info) in enumerate(sorted_colors[:3]):
        row = i
        if info['h_span'] > 2/3:
            output_values[row] = [color, color, color]
        elif info['v_span'] > 2/3:
            col = 2 if info['center'] > 2/3 else (1 if info['center'] > 1/3 else 0)
            output_values[row][col] = color
        else:
            col = 2 if info['center'] > 2/3 else (0 if info['center'] < 1/3 else 1)
            output_values[row][col] = color
    
    return output_values
