from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_8ba14f53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x9 input grid into a 3x3 output grid based on the following rules:
    1. Identify the three most prominent non-black colors in the input grid.
    2. For each color, determine its shape characteristics (extent, edge touching).
    3. Create a 3x3 output grid representing these colors' shapes simplified.
    4. Allocate rows based on color prominence and input shape.
    5. Fill cells in each row based on the color's characteristics in the input.
    6. Fill any remaining cells with black (0).

    The transformation preserves the relative positions and basic shape features
    of the most prominent colors while simplifying the overall representation.
    """
    # Step 1: Analyze the input grid
    colors = analyze_grid(input_grid)

    # Step 2: Sort colors by prominence
    sorted_colors = sorted(colors.items(), key=lambda x: (-x[1]['area'], x[1]['left'], x[1]['top']))

    # Step 3-6: Create and fill the output grid
    output_values = [[0, 0, 0] for _ in range(3)]
    row = 0
    for color, info in sorted_colors[:3]:
        if row >= 3:
            break
        fill_output_row(output_values, row, color, info)
        row += 1
        if row < 3 and info['area'] > sum(info['area'] for _, info in sorted_colors) / 4:
            fill_output_row(output_values, row, color, info)
            row += 1

    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> dict:
    colors = {}
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color != 0:
                if color not in colors:
                    colors[color] = {'left': c, 'top': r, 'right': c, 'bottom': r, 'area': 0}
                info = colors[color]
                info['left'] = min(info['left'], c)
                info['top'] = min(info['top'], r)
                info['right'] = max(info['right'], c)
                info['bottom'] = max(info['bottom'], r)
                info['area'] += 1
    return colors

def fill_output_row(output_values: List[List[int]], row: int, color: int, info: dict):
    rows, cols = 4, 9  # Assuming input is always 4x9
    output_values[row][0] = color if info['left'] == 0 else 0
    output_values[row][1] = color if (info['right'] - info['left']) > cols / 2 else 0
    output_values[row][2] = color if info['bottom'] == rows - 1 or (info['bottom'] - info['top']) > rows / 2 else 0
