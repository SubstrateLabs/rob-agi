from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import math

def solve_0a1d4ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a large input grid into a smaller output grid (2x2 or 3x3) by identifying
    and arranging the most visually significant colors.

    The solution follows these steps:
    1. Analyze the input grid to calculate visual significance of colors based on frequency,
       region size, contrast, and position.
    2. Determine the output grid size (2x2 or 3x3) based on the number of significant colors.
    3. Select the most visually significant colors for the output grid.
    4. Arrange the colors in the output grid to correspond with their positions and significance
       in the input grid while maintaining a balanced representation.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: A 2x2 or 3x3 grid representing the essence of the input grid.
    """
    # Step 1: Analyze the input grid
    color_significance = calculate_visual_significance(input_grid)
    
    # Step 2: Determine output grid size
    output_size = 2 if len(color_significance) <= 5 else 3
    
    # Step 3: Select colors for the output grid
    selected_colors = select_colors(color_significance, output_size * output_size)
    
    # Step 4: Arrange colors in the output grid
    output_values = arrange_colors(input_grid, selected_colors, output_size)

    return ColoredGrid(values=output_values)

def calculate_visual_significance(input_grid: ColoredGrid) -> Dict[int, float]:
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 0)
    color_regions = find_connected_regions(input_grid)
    rows, cols = input_grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2
    
    significance = {}
    for color, count in color_counts.items():
        largest_region = max(len(region) for region in color_regions[color])
        contrast = sum(abs(color - other_color) for other_color in color_counts if other_color != color)
        center_distance = min(math.dist((r, c), (center_row, center_col)) 
                              for region in color_regions[color] 
                              for r, c in region)
        
        significance[color] = (count * largest_region * contrast) / (center_distance + 1)
    
    return significance

def find_connected_regions(grid: ColoredGrid) -> Dict[int, List[List[Tuple[int, int]]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = defaultdict(list)

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc, color))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                region = dfs(r, c, color)
                if region:
                    regions[color].append(region)

    return regions

def select_colors(color_significance: Dict[int, float], num_colors: int) -> List[int]:
    return sorted(color_significance, key=color_significance.get, reverse=True)[:num_colors]

def arrange_colors(input_grid: ColoredGrid, selected_colors: List[int], output_size: int) -> List[List[int]]:
    rows, cols = input_grid.get_dimensions()
    output_grid = [[0 for _ in range(output_size)] for _ in range(output_size)]
    color_positions = {color: [] for color in selected_colors}

    for r in range(rows):
        for c in range(cols):
            color = input_grid.values[r][c]
            if color in selected_colors:
                color_positions[color].append((r / rows, c / cols))

    for color in selected_colors:
        if color_positions[color]:
            avg_r = sum(pos[0] for pos in color_positions[color]) / len(color_positions[color])
            avg_c = sum(pos[1] for pos in color_positions[color]) / len(color_positions[color])
            output_r = min(int(avg_r * output_size), output_size - 1)
            output_c = min(int(avg_c * output_size), output_size - 1)
            
            if output_grid[output_r][output_c] == 0:
                output_grid[output_r][output_c] = color
            else:
                # Find the nearest empty cell
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    nr, nc = output_r + dr, output_c + dc
                    if 0 <= nr < output_size and 0 <= nc < output_size and output_grid[nr][nc] == 0:
                        output_grid[nr][nc] = color
                        break
                else:
                    # If no empty cell found, replace the least significant color
                    for r in range(output_size):
                        for c in range(output_size):
                            if output_grid[r][c] == 0 or selected_colors.index(output_grid[r][c]) > selected_colors.index(color):
                                output_grid[r][c] = color
                                break
                        else:
                            continue
                        break

    # Fill any remaining empty cells
    empty_cells = [(r, c) for r in range(output_size) for c in range(output_size) if output_grid[r][c] == 0]
    for r, c in empty_cells:
        for color in selected_colors:
            if color not in [output_grid[i][j] for i in range(output_size) for j in range(output_size)]:
                output_grid[r][c] = color
                break
        else:
            output_grid[r][c] = selected_colors[0]  # Use the most significant color if all are already used

    return output_grid
