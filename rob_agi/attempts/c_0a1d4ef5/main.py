from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import math

def solve_0a1d4ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a large input grid into a smaller output grid (2x2 or 3x3) by identifying
    and arranging the most visually significant colors.

    The solution follows these steps:
    1. Analyze the input grid to calculate visual significance of colors based on region size,
       position, and contrast.
    2. Determine the output grid size (2x2 or 3x3) based on the complexity of the input.
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
    output_size = determine_output_size(input_grid, color_significance)
    
    # Step 3: Select colors for the output grid
    selected_colors = select_colors(color_significance, output_size * output_size)
    
    # Step 4: Arrange colors in the output grid
    output_values = arrange_colors(input_grid, selected_colors, output_size)

    return ColoredGrid(values=output_values)

def calculate_visual_significance(input_grid: ColoredGrid) -> Dict[int, float]:
    color_regions = find_connected_regions(input_grid)
    rows, cols = input_grid.get_dimensions()
    center_row, center_col = rows // 2, cols // 2
    
    significance = {}
    for color, regions in color_regions.items():
        total_area = sum(len(region) for region in regions)
        largest_region = max(len(region) for region in regions)
        center_of_mass = calculate_center_of_mass(regions)
        center_distance = math.dist(center_of_mass, (center_row, center_col))
        contrast = calculate_contrast(color, input_grid)
        
        significance[color] = (total_area * largest_region * contrast) / (center_distance + 1)
    
    return significance

def calculate_center_of_mass(regions: List[List[Tuple[int, int]]]) -> Tuple[float, float]:
    all_points = [point for region in regions for point in region]
    return (sum(p[0] for p in all_points) / len(all_points),
            sum(p[1] for p in all_points) / len(all_points))

def calculate_contrast(color: int, grid: ColoredGrid) -> float:
    other_colors = set(cell for row in grid.values for cell in row if cell != color and cell != 0)
    return sum(abs(color - other_color) for other_color in other_colors)

def determine_output_size(input_grid: ColoredGrid, color_significance: Dict[int, float]) -> int:
    significant_colors = sum(1 for score in color_significance.values() if score > max(color_significance.values()) * 0.1)
    complexity = len(set(cell for row in input_grid.values for cell in row if cell != 0))
    return 2 if significant_colors <= 4 or complexity <= 5 else 3

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
    color_regions = find_connected_regions(input_grid)
    
    # Calculate the center of mass for each color
    color_positions = {color: calculate_center_of_mass(regions) for color, regions in color_regions.items() if color in selected_colors}
    
    # Normalize positions to output grid size
    for color, (r, c) in color_positions.items():
        color_positions[color] = (r / rows * output_size, c / cols * output_size)
    
    # Place colors in the output grid
    for color in selected_colors:
        if color in color_positions:
            r, c = color_positions[color]
            output_r = min(int(r), output_size - 1)
            output_c = min(int(c), output_size - 1)
            
            if output_grid[output_r][output_c] == 0:
                output_grid[output_r][output_c] = color
            else:
                # Find the nearest empty cell
                best_distance = float('inf')
                best_pos = None
                for i in range(output_size):
                    for j in range(output_size):
                        if output_grid[i][j] == 0:
                            distance = math.dist((r, c), (i, j))
                            if distance < best_distance:
                                best_distance = distance
                                best_pos = (i, j)
                
                if best_pos:
                    output_grid[best_pos[0]][best_pos[1]] = color
                else:
                    # If no empty cell, replace the least significant color
                    least_significant = min(selected_colors, key=selected_colors.index)
                    for i in range(output_size):
                        for j in range(output_size):
                            if output_grid[i][j] == least_significant:
                                output_grid[i][j] = color
                                break
                        if output_grid[i][j] == color:
                            break
    
    # Fill any remaining empty cells
    empty_cells = [(r, c) for r in range(output_size) for c in range(output_size) if output_grid[r][c] == 0]
    for r, c in empty_cells:
        unused_colors = [color for color in selected_colors if color not in [output_grid[i][j] for i in range(output_size) for j in range(output_size)]]
        if unused_colors:
            output_grid[r][c] = unused_colors[0]
        else:
            output_grid[r][c] = selected_colors[0]  # Use the most significant color if all are already used

    return output_grid
