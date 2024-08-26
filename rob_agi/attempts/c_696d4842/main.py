from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_696d4842(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a series of operations:
    1. Identify and categorize colored elements.
    2. Extend vertical and horizontal lines.
    3. Complete partial shapes into rectangles or larger L-shapes.
    4. Transform colors based on surrounding context.
    5. Balance the composition by extending or introducing new elements.
    6. Resolve conflicts between extending elements.
    7. Fine-tune by smoothing edges and filling small gaps.
    8. Iterate until the grid stabilizes or reaches a maximum number of iterations.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    color_cycle = {4: 2, 2: 6, 6: 1, 1: 3, 3: 8, 8: 4}  # Yellow -> Red -> Magenta -> Blue -> Green -> Sky -> Yellow

    def identify_elements():
        elements = {'vertical': [], 'horizontal': [], 'l_shape': [], 'isolated': []}
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] != 0:
                    if r > 0 and output_grid.values[r-1][c] == output_grid.values[r][c]:
                        elements['vertical'].append((r, c))
                    elif c > 0 and output_grid.values[r][c-1] == output_grid.values[r][c]:
                        elements['horizontal'].append((r, c))
                    elif (r > 0 and output_grid.values[r-1][c] == output_grid.values[r][c]) and \
                         (c > 0 and output_grid.values[r][c-1] == output_grid.values[r][c]):
                        elements['l_shape'].append((r, c))
                    else:
                        elements['isolated'].append((r, c))
        return elements

    def extend_lines(elements):
        for r, c in elements['vertical']:
            color = output_grid.values[r][c]
            # Extend upwards
            for up in range(r-1, -1, -1):
                if output_grid.values[up][c] == 0:
                    output_grid.values[up][c] = color if up > 1 else color_cycle[color]
                else:
                    break
            # Extend downwards
            for down in range(r+1, rows):
                if output_grid.values[down][c] == 0:
                    output_grid.values[down][c] = color
                else:
                    break
        
        for r, c in elements['horizontal']:
            color = output_grid.values[r][c]
            # Extend left
            for left in range(c-1, -1, -1):
                if output_grid.values[r][left] == 0:
                    output_grid.values[r][left] = color
                else:
                    break
            # Extend right
            for right in range(c+1, cols):
                if output_grid.values[r][right] == 0:
                    output_grid.values[r][right] = color
                else:
                    break

    def complete_shapes(elements):
        for r, c in elements['l_shape']:
            color = output_grid.values[r][c]
            # Complete rectangle
            max_width = max_height = 0
            for width in range(c, cols):
                if output_grid.values[r][width] != color:
                    break
                max_width = width - c + 1
            for height in range(r, rows):
                if output_grid.values[height][c] != color:
                    break
                max_height = height - r + 1
            for i in range(r, r + max_height):
                for j in range(c, c + max_width):
                    output_grid.values[i][j] = color

    def transform_colors():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] != 0:
                    neighbors = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                                 if 0 <= r+dr < rows and 0 <= c+dc < cols]
                    neighbor_colors = [output_grid.values[nr][nc] for nr, nc in neighbors if output_grid.values[nr][nc] != 0]
                    if neighbor_colors:
                        most_common = max(set(neighbor_colors), key=neighbor_colors.count)
                        if most_common in color_cycle:
                            output_grid.values[r][c] = color_cycle[most_common]

    def balance_composition(elements):
        total_colored = sum(len(v) for v in elements.values())
        target_colored = rows * cols // 3  # Aim for about 1/3 of the grid to be colored
        if total_colored < target_colored:
            empty_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 0]
            for r, c in empty_cells[:target_colored - total_colored]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                neighbor_colors = [output_grid.values[nr][nc] for nr, nc in neighbors if output_grid.values[nr][nc] != 0]
                if neighbor_colors:
                    output_grid.values[r][c] = max(set(neighbor_colors), key=neighbor_colors.count)

    def smooth_edges():
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if output_grid.values[r][c] == 0:
                    neighbors = [output_grid.values[r+dr][c+dc] for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
                    non_zero = [color for color in neighbors if color != 0]
                    if len(non_zero) >= 3:
                        output_grid.values[r][c] = max(set(non_zero), key=non_zero.count)

    iterations = 0
    while iterations < 10:  # Increased max iterations
        old_grid = output_grid.deep_copy()
        elements = identify_elements()
        extend_lines(elements)
        complete_shapes(elements)
        transform_colors()
        balance_composition(elements)
        smooth_edges()
        if output_grid.values == old_grid.values:
            break
        iterations += 1

    return output_grid
