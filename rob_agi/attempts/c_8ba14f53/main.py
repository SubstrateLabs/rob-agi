from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple

def solve_8ba14f53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x9 input grid into a 3x3 output grid based on the following rules:
    1. Identify the three most prominent non-black colors in the input grid.
    2. Analyze each color's shape characteristics (extent, edge touching, density).
    3. Rank colors based on prominence, density, and vertical position.
    4. Allocate rows in the output grid based on color ranking and input shape.
    5. Represent each color in its allocated row(s) based on its characteristics.
    6. Fill any remaining cells with black (0).

    The transformation preserves the relative positions and basic shape features
    of the most prominent colors while simplifying the overall representation.
    """
    # Step 1-2: Analyze the input grid
    colors = analyze_grid(input_grid)

    # Step 3: Rank colors
    sorted_colors = rank_colors(colors)

    # Step 4-6: Create and fill the output grid
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
                    colors[color] = {'left': c, 'top': r, 'right': c, 'bottom': r, 'area': 0, 'count': 0}
                info = colors[color]
                info['left'] = min(info['left'], c)
                info['top'] = min(info['top'], r)
                info['right'] = max(info['right'], c)
                info['bottom'] = max(info['bottom'], r)
                info['count'] += 1
    
    for color, info in colors.items():
        info['area'] = (info['right'] - info['left'] + 1) * (info['bottom'] - info['top'] + 1)
        info['density'] = info['count'] / info['area']
        info['h_span'] = (info['right'] - info['left'] + 1) / cols
        info['v_span'] = (info['bottom'] - info['top'] + 1) / rows
    
    return colors

def rank_colors(colors: Dict[int, Dict]) -> List[Tuple[int, Dict]]:
    return sorted(colors.items(), key=lambda x: (-x[1]['count'], -x[1]['density'], x[1]['top']))

def create_output_grid(sorted_colors: List[Tuple[int, Dict]], input_dims: Tuple[int, int]) -> List[List[int]]:
    output_values = [[0, 0, 0] for _ in range(3)]
    rows, cols = input_dims
    total_area = sum(info['count'] for _, info in sorted_colors)
    
    for i, (color, info) in enumerate(sorted_colors[:3]):
        row = i
        if i == 0 and info['count'] > total_area / 3:
            fill_output_row(output_values, row, color, info, rows, cols)
            row += 1
            if row < 3:
                fill_output_row(output_values, row, color, info, rows, cols)
        else:
            fill_output_row(output_values, row, color, info, rows, cols)
    
    return output_values

def fill_output_row(output_values: List[List[int]], row: int, color: int, info: Dict, input_rows: int, input_cols: int):
    if info['h_span'] > 2/3:
        output_values[row] = [color, color, color]
    elif info['v_span'] > 2/3:
        col = 1 if info['left'] + info['right'] > input_cols else 0
        output_values[row][col] = color
    else:
        if info['left'] == 0:
            output_values[row][0] = color
        if info['right'] == input_cols - 1:
            output_values[row][2] = color
        if output_values[row][0] == 0 and output_values[row][2] == 0:
            output_values[row][1] = color
