from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_642d658d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 642d658d challenge by identifying the most significant non-background pattern in the input grid.
    
    The solution follows these steps:
    1. Identify the dominant (background) color
    2. For each non-dominant color, analyze its connected components
    3. Score each color based on the characteristics of its components
    4. Select the highest-scoring color as the most significant pattern
    5. Return a 1x1 grid with the selected color
    
    This approach captures the essence of the pattern across all examples by focusing on
    the most structurally significant non-background elements in the grid.
    """
    # Count color occurrences
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    background_color = max(color_counts, key=color_counts.get)
    
    # Analyze non-background colors
    color_scores = {}
    for color in set(color_counts.keys()) - {background_color}:
        components = find_connected_components(input_grid, color)
        score = score_color(components, input_grid.get_dimensions())
        color_scores[color] = score
    
    # Select the highest-scoring color
    if color_scores:
        selected_color = max(color_scores, key=color_scores.get)
    else:
        selected_color = background_color  # Fallback if no non-background colors
    
    # Return a 1x1 grid with the selected color
    return ColoredGrid(values=[[selected_color]])

def find_connected_components(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected components of a given color in the grid."""
    return grid.find_connected_regions(color)

def score_color(components: List[List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> float:
    """Score a color based on its components' characteristics."""
    if not components:
        return 0
    
    total_size = sum(len(comp) for comp in components)
    num_components = len(components)
    avg_size = total_size / num_components
    
    # Simple scoring based on number of components and their average size
    # This can be expanded to include more sophisticated shape and pattern analysis
    score = num_components * avg_size / (grid_dimensions[0] * grid_dimensions[1])
    
    return score
