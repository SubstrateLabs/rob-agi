from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import random

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing color patterns and distributions.
    
    The transformation process involves:
    1. Analyzing global and quadrant-specific color distributions in the input grid.
    2. Creating an initial 5x5 output grid based on 2x2 subgrids of the input.
    3. Applying a smoothing pass to ensure color consistency.
    4. Balancing color distribution based on input grid frequencies.
    5. Handling black (0) cells appropriately.
    6. Performing a final consistency check on 2x2 regions of the output.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    def analyze_colors(grid: List[List[int]], start_row: int, start_col: int, size: int) -> Counter:
        return Counter(grid[r][c] for r in range(start_row, start_row + size) 
                       for c in range(start_col, start_col + size))

    global_colors = analyze_colors(input_grid.values, 0, 0, 10)
    quadrant_colors = [
        analyze_colors(input_grid.values, r, c, 5)
        for r in [0, 5] for c in [0, 5]
    ]

    def get_dominant_color(colors: Counter, exclude: set = set()) -> int:
        return max((c for c in colors.items() if c[0] not in exclude), key=lambda x: x[1])[0]

    def process_region(region: List[List[int]], quad_index: int) -> int:
        local_colors = Counter(cell for row in region for cell in row if cell != 0)
        if not local_colors:
            return get_dominant_color(quadrant_colors[quad_index], exclude={0})
        return get_dominant_color(local_colors)

    output_values = []
    for i in range(5):
        row = []
        for j in range(5):
            region = [input_grid.values[i*2+di][j*2:j*2+2] for di in range(2)]
            quad_index = (i // 3) * 2 + (j // 3)
            color = process_region(region, quad_index)
            row.append(color)
        output_values.append(row)

    def smooth_output(values: List[List[int]]) -> List[List[int]]:
        new_values = [row[:] for row in values]
        for i in range(5):
            for j in range(5):
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
                valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5]
                neighbor_colors = Counter(values[r][c] for r, c in valid_neighbors)
                if values[i][j] not in neighbor_colors:
                    new_values[i][j] = get_dominant_color(neighbor_colors, exclude={0})
        return new_values

    output_values = smooth_output(output_values)

    def balance_colors(values: List[List[int]]) -> List[List[int]]:
        output_colors = Counter(color for row in values for color in row)
        for color, count in global_colors.most_common():
            target = round(count * 25 / 100)
            while output_colors[color] < target:
                i, j = random.randint(0, 4), random.randint(0, 4)
                if values[i][j] != color and output_colors[values[i][j]] > target:
                    output_colors[values[i][j]] -= 1
                    values[i][j] = color
                    output_colors[color] += 1
        return values

    output_values = balance_colors(output_values)

    black_ratio = global_colors[0] / 100
    if black_ratio > 0.25 and 0 not in (color for row in output_values for color in row):
        i, j = random.randint(0, 4), random.randint(0, 4)
        output_values[i][j] = 0
    elif black_ratio > 0.5:
        for _ in range(2):
            i, j = random.randint(0, 4), random.randint(0, 4)
            output_values[i][j] = 0

    def ensure_consistency(values: List[List[int]]) -> List[List[int]]:
        for i in range(4):
            for j in range(4):
                region = [values[i+di][j:j+2] for di in range(2)]
                colors = set(cell for row in region for cell in row)
                if len(colors) == 4:
                    least_common = min(colors, key=lambda c: sum(row.count(c) for row in values))
                    values[i+random.randint(0,1)][j+random.randint(0,1)] = get_dominant_color(Counter(cell for row in region for cell in row))
        return values

    output_values = ensure_consistency(output_values)

    return ColoredGrid(values=output_values)
