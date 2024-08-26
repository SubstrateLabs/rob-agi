from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict

def solve_642d658d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 642d658d challenge by identifying the most significant pattern in the input grid.
    
    The solution follows these steps:
    1. Identify the background color
    2. Analyze non-background colors and their relationships
    3. Score colors based on their characteristics and relationships
    4. Select the highest-scoring color as the most significant pattern
    5. Return a 1x1 grid with the selected color
    
    This approach captures complex patterns by considering color distributions,
    shapes, and relationships between different colors in the grid.
    """
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    background_color = max(color_counts, key=color_counts.get)
    
    non_background_colors = set(color_counts.keys()) - {background_color}
    color_components = {color: find_connected_components(input_grid, color) for color in non_background_colors}
    
    color_scores = score_colors(color_components, input_grid.get_dimensions(), background_color)
    
    if color_scores:
        selected_color = max(color_scores, key=color_scores.get)
    else:
        selected_color = background_color  # Fallback if no non-background colors
    
    return ColoredGrid(values=[[selected_color]])

def find_connected_components(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected components of a given color in the grid."""
    return grid.find_connected_regions(color)

def score_colors(color_components: Dict[int, List[List[Tuple[int, int]]]], 
                 grid_dimensions: Tuple[int, int], 
                 background_color: int) -> Dict[int, float]:
    """Score colors based on their components' characteristics and relationships."""
    scores = {}
    total_area = grid_dimensions[0] * grid_dimensions[1]
    
    for color, components in color_components.items():
        if not components:
            continue
        
        total_size = sum(len(comp) for comp in components)
        num_components = len(components)
        avg_size = total_size / num_components
        
        # Score based on coverage, number of components, and shape complexity
        coverage_score = total_size / total_area
        component_score = min(num_components / 10, 1)  # Cap at 1 for grids with many small components
        shape_score = calculate_shape_score(components, grid_dimensions)
        
        # Combine scores with weights
        scores[color] = (coverage_score * 0.4 + component_score * 0.3 + shape_score * 0.3) * (color != background_color)
    
    return scores

def calculate_shape_score(components: List[List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score based on the shapes of the components."""
    total_perimeter = sum(calculate_perimeter(comp) for comp in components)
    total_area = sum(len(comp) for comp in components)
    
    if total_area == 0:
        return 0
    
    # A perfect square would have a perimeter-to-area ratio of 4/sqrt(area)
    # Higher ratios indicate more complex shapes
    shape_complexity = (total_perimeter / total_area) / (4 / (total_area ** 0.5))
    
    return min(shape_complexity / 2, 1)  # Cap at 1 for very complex shapes

def calculate_perimeter(component: List[Tuple[int, int]]) -> int:
    """Calculate the perimeter of a component."""
    perimeter = 0
    component_set = set(component)
    for x, y in component:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if (x + dx, y + dy) not in component_set:
                perimeter += 1
    return perimeter
