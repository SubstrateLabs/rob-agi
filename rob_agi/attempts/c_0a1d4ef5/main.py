from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import math

def solve_0a1d4ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a large input grid into a smaller output grid (2x2, 2x3, or 3x3) by identifying
    and arranging the most visually significant colors.

    The solution follows these steps:
    1. Analyze the input grid to calculate visual significance of colors based on region size,
       position, contrast, connectivity, and other factors.
    2. Determine the output grid size (2x2, 2x3, or 3x3) based on the number of significant colors.
    3. Select the most visually significant colors for the output grid.
    4. Create a color relationship map to understand the relative positions and adjacencies of colors.
    5. Arrange the colors in the output grid to best represent their relationships and significance.
    6. Fine-tune the arrangement to maximize the representation of the input grid's essence.
    7. Handle edge cases and ensure the output is always valid.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: A 2x2, 2x3, or 3x3 grid representing the essence of the input grid.
    """
    # Step 1: Analyze the input grid
    color_significance = calculate_visual_significance(input_grid)
    
    # Step 2: Determine output grid size
    output_size = determine_output_size(color_significance)
    
    # Step 3: Select colors for the output grid
    selected_colors = select_colors(color_significance, output_size[0] * output_size[1])
    
    # Step 4: Create color relationship map
    color_relationships = create_color_relationship_map(input_grid, selected_colors)
    
    # Step 5: Arrange colors in the output grid
    output_values = arrange_colors(selected_colors, color_relationships, output_size)
    
    # Step 6: Fine-tune the arrangement
    output_values = fine_tune_arrangement(output_values, color_relationships)

    # Step 7: Handle edge cases
    output_values = handle_edge_cases(output_values, selected_colors)

    return ColoredGrid(values=output_values)

def calculate_visual_significance(input_grid: ColoredGrid) -> Dict[int, float]:
    color_regions = find_connected_regions(input_grid)
    rows, cols = input_grid.get_dimensions()
    center_row, center_col = rows / 2, cols / 2
    
    significance = {}
    for color, regions in color_regions.items():
        total_area = sum(len(region) for region in regions)
        largest_region = max(len(region) for region in regions)
        center_of_mass = calculate_center_of_mass(regions)
        center_distance = math.dist(center_of_mass, (center_row, center_col))
        contrast = calculate_contrast(color, input_grid)
        connectivity = len(regions)
        
        significance[color] = (total_area * largest_region * contrast * connectivity) / (center_distance + 1)
    
    # Normalize significance scores
    max_significance = max(significance.values())
    for color in significance:
        significance[color] /= max_significance
    
    return significance

def calculate_center_of_mass(regions: List[List[Tuple[int, int]]]) -> Tuple[float, float]:
    all_points = [point for region in regions for point in region]
    return (sum(p[0] for p in all_points) / len(all_points),
            sum(p[1] for p in all_points) / len(all_points))

def calculate_contrast(color: int, grid: ColoredGrid) -> float:
    other_colors = set(cell for row in grid.values for cell in row if cell != color and cell != 0)
    return sum(abs(color - other_color) for other_color in other_colors)

def determine_output_size(color_significance: Dict[int, float]) -> Tuple[int, int]:
    significant_colors = sum(1 for score in color_significance.values() if score > 0.1)
    if significant_colors <= 4:
        return (2, 2)
    elif significant_colors <= 6:
        return (2, 3)
    else:
        return (3, 3)

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

def create_color_relationship_map(input_grid: ColoredGrid, selected_colors: List[int]) -> Dict[Tuple[int, int], float]:
    relationships = {}
    color_positions = {color: calculate_center_of_mass(find_connected_regions(input_grid)[color]) 
                       for color in selected_colors}
    
    for color1 in selected_colors:
        for color2 in selected_colors:
            if color1 != color2:
                pos1 = color_positions[color1]
                pos2 = color_positions[color2]
                relationships[(color1, color2)] = math.dist(pos1, pos2)
    
    return relationships

def arrange_colors(selected_colors: List[int], color_relationships: Dict[Tuple[int, int], float], output_size: Tuple[int, int]) -> List[List[int]]:
    rows, cols = output_size
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Place the most significant color in the center
    center_r, center_c = rows // 2, cols // 2
    output_grid[center_r][center_c] = selected_colors[0]
    
    # Place other colors
    for color in selected_colors[1:]:
        best_position = None
        best_score = float('inf')
        for r in range(rows):
            for c in range(cols):
                if output_grid[r][c] == 0:
                    score = sum(abs(color_relationships.get((color, output_grid[pr][pc]), 0) - 
                                    math.dist((r, c), (pr, pc)))
                                for pr in range(rows) for pc in range(cols)
                                if output_grid[pr][pc] != 0)
                    if score < best_score:
                        best_score = score
                        best_position = (r, c)
        
        if best_position:
            output_grid[best_position[0]][best_position[1]] = color
    
    return output_grid

def fine_tune_arrangement(output_grid: List[List[int]], color_relationships: Dict[Tuple[int, int], float]) -> List[List[int]]:
    rows, cols = len(output_grid), len(output_grid[0])
    best_grid = output_grid
    best_score = calculate_arrangement_score(output_grid, color_relationships)
    
    for _ in range(100):  # Try 100 iterations of swapping
        r1, c1 = random.randint(0, rows-1), random.randint(0, cols-1)
        r2, c2 = random.randint(0, rows-1), random.randint(0, cols-1)
        new_grid = [row[:] for row in output_grid]
        new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]
        new_score = calculate_arrangement_score(new_grid, color_relationships)
        
        if new_score < best_score:
            best_grid = new_grid
            best_score = new_score
    
    return best_grid

def calculate_arrangement_score(grid: List[List[int]], color_relationships: Dict[Tuple[int, int], float]) -> float:
    score = 0
    rows, cols = len(grid), len(grid[0])
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(rows):
                for c2 in range(cols):
                    if r1 != r2 or c1 != c2:
                        color1, color2 = grid[r1][c1], grid[r2][c2]
                        actual_distance = math.dist((r1, c1), (r2, c2))
                        ideal_distance = color_relationships.get((color1, color2), 0)
                        score += abs(actual_distance - ideal_distance)
    return score
def handle_edge_cases(output_values: List[List[int]], selected_colors: List[int]) -> List[List[int]]:
    rows, cols = len(output_values), len(output_values[0])
    total_cells = rows * cols
    
    # If there are less colors than cells, fill the remaining cells with the most significant color
    if len(selected_colors) < total_cells:
        for r in range(rows):
            for c in range(cols):
                if output_values[r][c] == 0:
                    output_values[r][c] = selected_colors[0]
    
    # Ensure at least one cell has the most significant color
    if selected_colors[0] not in [color for row in output_values for color in row]:
        output_values[0][0] = selected_colors[0]
    
    return output_values
