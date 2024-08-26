from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import random

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing color patterns and distributions.
    
    The transformation process involves:
    1. Analyzing global and quadrant-specific color distributions in the input grid.
    2. Identifying dominant colors and their positions.
    3. Generating an initial output grid based on local and global color information.
    4. Applying spatial relationship rules to maintain patterns.
    5. Handling black (0) cells appropriately.
    6. Balancing color distribution and adding controlled randomness.
    7. Performing a final coherence check and adjustment.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    def analyze_colors(grid: ColoredGrid, start_row: int, start_col: int, size: int) -> Counter:
        return Counter(grid.values[r][c] for r in range(start_row, start_row + size) 
                       for c in range(start_col, start_col + size))

    global_colors = analyze_colors(input_grid, 0, 0, 10)
    quadrant_colors = [
        analyze_colors(input_grid, r, c, 5)
        for r in [0, 5] for c in [0, 5]
    ]

    def get_dominant_color(colors: Counter, exclude: set = set()) -> int:
        return max((c for c in colors.items() if c[0] not in exclude), key=lambda x: x[1])[0]

    def process_region(region: List[List[int]], quad_index: int) -> int:
        local_colors = Counter(cell for row in region for cell in row)
        if local_colors[0] >= 3:  # If majority is black, keep it black
            return 0
        quad_dominant = get_dominant_color(quadrant_colors[quad_index], exclude={0})
        local_dominant = get_dominant_color(local_colors, exclude={0})
        return local_dominant if local_colors[local_dominant] >= 2 else quad_dominant

    output_values = []
    for i in range(5):
        row = []
        for j in range(5):
            region = [input_grid.values[i*2+di][j*2:j*2+2] for di in range(2)]
            quad_index = (i // 3) * 2 + (j // 3)
            color = process_region(region, quad_index)
            row.append(color)
        output_values.append(row)

    def adjust_output(output_values: List[List[int]]) -> List[List[int]]:
        for i in range(5):
            for j in range(5):
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5]
                neighbor_colors = [output_values[r][c] for r, c in valid_neighbors]
                
                if output_values[i][j] == 0 and any(neighbor_colors):
                    output_values[i][j] = max(set(neighbor_colors) - {0}, key=neighbor_colors.count)
                
                elif random.random() < 0.2:  # Introduce controlled randomness
                    quad_index = (i // 3) * 2 + (j // 3)
                    possible_colors = [c for c, count in quadrant_colors[quad_index].items() if count > 1 and c != 0]
                    if possible_colors:
                        output_values[i][j] = random.choice(possible_colors)

        return output_values

    output_values = adjust_output(output_values)

    # Balance check and adjustment
    output_colors = Counter(color for row in output_values for color in row)
    for color, count in global_colors.most_common(3):
        if color != 0 and output_colors[color] < count // 4:
            for _ in range(2):
                i, j = random.randint(0, 4), random.randint(0, 4)
                output_values[i][j] = color

    # Final coherence check
    for i in range(5):
        for j in range(5):
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5]
            neighbor_colors = [output_values[r][c] for r, c in valid_neighbors]
            if output_values[i][j] not in neighbor_colors and len(set(neighbor_colors)) > 1:
                output_values[i][j] = max(set(neighbor_colors), key=neighbor_colors.count)

    return ColoredGrid(values=output_values)
