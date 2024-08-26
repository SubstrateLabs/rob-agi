from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by analyzing color patterns and distributions.
    
    The transformation process involves:
    1. Analyzing global color distribution in the input grid.
    2. Processing 4x4 regions of the input grid to determine each output cell.
    3. Balancing local color information with global patterns and distribution.
    4. Preserving significant color transitions and patterns.
    5. Fine-tuning the output to maintain overall color balance and coherence.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    def analyze_global_colors(grid: ColoredGrid) -> Counter:
        return Counter(cell for row in grid.values for cell in row)

    def color_energy(region: List[List[int]]) -> Counter:
        weights = {0: 0, 1: 2, 2: 2, 3: 3, 4: 2, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
        return Counter({color: count * weights[color] for color, count in Counter(cell for row in region for cell in row).items()})

    def process_region(region: List[List[int]], global_colors: Counter) -> int:
        energy = color_energy(region)
        if not energy:
            return 0
        
        top_colors = energy.most_common(3)
        
        # Preserve significant colors
        if top_colors[0][1] > 1.5 * (top_colors[1][1] if len(top_colors) > 1 else 0):
            return top_colors[0][0]
        
        # Balance with global distribution
        for color, _ in top_colors:
            if global_colors[color] > 10:
                return color
        
        return top_colors[0][0]

    global_colors = analyze_global_colors(input_grid)
    output_values = []

    for i in range(0, 10, 2):
        row = []
        for j in range(0, 10, 2):
            region = [input_grid.values[i+di][j:j+2] for di in range(2)]
            color = process_region(region, global_colors)
            row.append(color)
        output_values.append(row)

    # Fine-tune output
    def adjust_output(output_values: List[List[int]], global_colors: Counter) -> List[List[int]]:
        output_colors = Counter(cell for row in output_values for cell in row)
        for i in range(5):
            for j in range(5):
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                valid_neighbors = [(r, c) for r, c in neighbors if 0 <= r < 5 and 0 <= c < 5]
                neighbor_colors = Counter(output_values[r][c] for r, c in valid_neighbors)
                
                current_color = output_values[i][j]
                if output_colors[current_color] > global_colors[current_color] * 0.3:
                    for color in range(10):
                        if output_colors[color] < global_colors[color] * 0.2 and color not in neighbor_colors:
                            output_values[i][j] = color
                            output_colors[current_color] -= 1
                            output_colors[color] += 1
                            break
        return output_values

    output_values = adjust_output(output_values, global_colors)

    return ColoredGrid(values=output_values)
