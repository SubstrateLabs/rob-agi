from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple
import random
import math

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing color patterns and distributions.
    
    The transformation process involves:
    1. Analyzing global, quadrant-specific, and local color distributions in the input grid.
    2. Calculating color importance based on frequency and context.
    3. Creating a color influence map to consider neighboring effects.
    4. Generating an initial output grid based on local and global color information.
    5. Applying context-aware adjustments to improve coherence.
    6. Implementing color transitions and mixing to represent significant boundaries.
    7. Balancing the overall color distribution.
    8. Introducing controlled randomness for a more natural feel.
    9. Performing a final coherence check and adjustment.
    
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

    def color_importance(color: int, count: int, quadrant_count: int) -> float:
        weights = {0: 0.1, 1: 1, 2: 1, 3: 1.2, 4: 1.1, 5: 0.8, 6: 1, 7: 1, 8: 1.1, 9: 1.2}
        global_importance = count / sum(global_colors.values())
        quadrant_importance = quadrant_count / sum(quadrant_colors[quad_index].values())
        return (global_importance + quadrant_importance) * weights[color]

    def create_influence_map(input_grid: ColoredGrid) -> List[List[List[float]]]:
        influence_map = [[[0.0 for _ in range(10)] for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                for di in range(2):
                    for dj in range(2):
                        r, c = i*2 + di, j*2 + dj
                        color = input_grid.values[r][c]
                        for ni in range(max(0, i-1), min(5, i+2)):
                            for nj in range(max(0, j-1), min(5, j+2)):
                                distance = math.sqrt((ni-i)**2 + (nj-j)**2)
                                influence = 1 / (1 + distance)
                                influence_map[ni][nj][color] += influence
        return influence_map

    influence_map = create_influence_map(input_grid)

    def process_region(region: List[List[int]], quad_index: int, i: int, j: int) -> int:
        local_colors = Counter(cell for row in region for cell in row)
        color_scores = {}
        for color, count in local_colors.items():
            quadrant_count = quadrant_colors[quad_index][color]
            importance = color_importance(color, global_colors[color], quadrant_count)
            influence = influence_map[i][j][color]
            color_scores[color] = count * importance * (1 + influence)
        
        best_color = max(color_scores, key=color_scores.get)
        if best_color == 0 and sum(local_colors.values()) > 1:
            non_zero_colors = [c for c in local_colors if c != 0]
            return random.choice(non_zero_colors) if non_zero_colors else 0
        return best_color

    output_values = []
    for i in range(5):
        row = []
        for j in range(5):
            region = [input_grid.values[i*2+di][j*2:j*2+2] for di in range(2)]
            quad_index = (i // 3) * 2 + (j // 3)
            color = process_region(region, quad_index, i, j)
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
                
                elif random.random() < 0.15:  # Introduce controlled randomness
                    quad_index = (i // 3) * 2 + (j // 3)
                    possible_colors = [c for c, count in quadrant_colors[quad_index].items() if count > 1 and c != 0]
                    if possible_colors:
                        output_values[i][j] = random.choice(possible_colors)

        return output_values

    output_values = adjust_output(output_values)

    # Color transition and mixing
    for i in range(5):
        for j in range(5):
            if i < 4 and output_values[i][j] != output_values[i+1][j]:
                if random.random() < 0.5:
                    output_values[i][j] = output_values[i+1][j]
            if j < 4 and output_values[i][j] != output_values[i][j+1]:
                if random.random() < 0.5:
                    output_values[i][j] = output_values[i][j+1]

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
