from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import random

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing color patterns and distributions.
    
    The transformation process involves:
    1. Analyzing global and quadrant-specific color distributions in the input grid.
    2. Processing 2x2 regions of the input grid to determine each output cell.
    3. Balancing local color information with global patterns and distribution.
    4. Preserving significant color transitions and patterns.
    5. Introducing controlled randomness to create a more mixed and intuitive feel.
    6. Fine-tuning the output to maintain overall color balance and coherence.
    7. Adjusting edges and corners to better represent the input grid's characteristics.
    
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

    def color_importance(color: int, count: int) -> float:
        weights = {0: 0.1, 1: 1, 2: 1, 3: 1.2, 4: 1, 5: 0.8, 6: 1, 7: 1, 8: 1, 9: 1}
        return count * weights[color]

    def process_region(region: List[List[int]], quad_index: int) -> int:
        local_colors = Counter(cell for row in region for cell in row)
        quad_color_importance = {color: color_importance(color, count) 
                                 for color, count in quadrant_colors[quad_index].items()}
        
        best_color = max(local_colors, key=lambda c: (
            local_colors[c] * color_importance(c, global_colors[c]) * quad_color_importance.get(c, 0)
        ))
        
        if best_color == 0 and sum(local_colors.values()) > 1:
            non_zero_colors = [c for c in local_colors if c != 0]
            return random.choice(non_zero_colors) if non_zero_colors else 0
        
        return best_color

    output_values = []
    for i in range(0, 10, 2):
        row = []
        for j in range(0, 10, 2):
            region = [input_grid.values[i+di][j:j+2] for di in range(2)]
            quad_index = (i // 5) * 2 + (j // 5)
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
                    output_values[i][j] = random.choice([c for c in neighbor_colors if c != 0])
                
                elif random.random() < 0.2:  # Introduce controlled randomness
                    quad_index = (i // 3) * 2 + (j // 3)
                    possible_colors = [c for c, count in quadrant_colors[quad_index].items() if count > 1 and c != 0]
                    if possible_colors:
                        output_values[i][j] = random.choice(possible_colors)

        return output_values

    output_values = adjust_output(output_values)

    # Adjust edges and corners
    for i in [0, 4]:
        for j in [0, 4]:
            corner_region = [input_grid.values[r][c] for r in range(i*2, i*2+2) for c in range(j*2, j*2+2)]
            corner_colors = Counter(corner_region)
            if corner_colors:
                output_values[i][j] = max(corner_colors, key=corner_colors.get)

    return ColoredGrid(values=output_values)
