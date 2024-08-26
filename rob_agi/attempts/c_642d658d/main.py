from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import math

def solve_642d658d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 642d658d challenge by identifying the most significant pattern in the input grid.
    
    The solution follows these steps:
    1. Analyze the background and identify non-background colors
    2. Find connected components for each non-background color
    3. Calculate various significance scores for each color:
       - Coverage
       - Component count and sizes
       - Shape complexity
       - Contrast with background
       - Pattern/symmetry
       - Multi-scale significance
       - Contextual significance (centrality)
       - Color relationships
       - Structural importance
    4. Compute overall significance scores
    5. Select the color with the highest significance score
    6. Return a 1x1 grid with the selected color
    
    This approach captures complex patterns by considering multiple aspects of color significance,
    from low-level details to high-level patterns, allowing for a comprehensive analysis of the input grid.
    """
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    background_color = max(color_counts, key=color_counts.get)
    total_cells = sum(color_counts.values())
    
    non_background_colors = set(color_counts.keys()) - {background_color}
    color_components = {color: find_connected_components(input_grid, color) for color in non_background_colors}
    
    color_scores = calculate_color_scores(input_grid, color_components, background_color, total_cells)
    
    if color_scores:
        selected_color = max(color_scores, key=color_scores.get)
    else:
        selected_color = background_color  # Fallback if no non-background colors
    
    return ColoredGrid(values=[[selected_color]])

def find_connected_components(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected components of a given color in the grid."""
    return grid.find_connected_regions(color)

def calculate_color_scores(grid: ColoredGrid, color_components: Dict[int, List[List[Tuple[int, int]]]], 
                           background_color: int, total_cells: int) -> Dict[int, float]:
    """Calculate comprehensive significance scores for each color."""
    scores = {}
    grid_dimensions = grid.get_dimensions()
    
    for color, components in color_components.items():
        if not components:
            continue
        
        coverage_score = sum(len(comp) for comp in components) / total_cells
        component_score = calculate_component_score(components)
        shape_score = calculate_shape_score(components, grid_dimensions)
        contrast_score = calculate_contrast_score(color, background_color)
        pattern_score = calculate_pattern_score(components, grid_dimensions)
        multi_scale_score = calculate_multi_scale_score(grid, color, grid_dimensions)
        centrality_score = calculate_centrality_score(components, grid_dimensions)
        relationship_score = calculate_relationship_score(grid, color, background_color)
        structural_score = calculate_structural_score(grid, color, background_color)
        
        # Combine scores with weights
        scores[color] = (
            coverage_score * 0.15 +
            component_score * 0.1 +
            shape_score * 0.1 +
            contrast_score * 0.1 +
            pattern_score * 0.15 +
            multi_scale_score * 0.1 +
            centrality_score * 0.1 +
            relationship_score * 0.1 +
            structural_score * 0.1
        )
    
    return scores

def calculate_component_score(components: List[List[Tuple[int, int]]]) -> float:
    """Calculate a score based on the number and size of components."""
    num_components = len(components)
    avg_size = sum(len(comp) for comp in components) / num_components if num_components > 0 else 0
    return min((num_components * avg_size) / 100, 1)  # Normalize and cap at 1

def calculate_shape_score(components: List[List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score based on the shapes of the components."""
    total_perimeter = sum(calculate_perimeter(comp) for comp in components)
    total_area = sum(len(comp) for comp in components)
    
    if total_area == 0:
        return 0
    
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

def calculate_contrast_score(color: int, background_color: int) -> float:
    """Calculate a contrast score between a color and the background color."""
    return abs(color - background_color) / 9  # Normalize by max color difference

def calculate_pattern_score(components: List[List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score based on repeating patterns and symmetries."""
    pattern_score = 0
    for component in components:
        # Check for symmetry
        symmetry_score = calculate_symmetry_score(component, grid_dimensions)
        # Check for repeating patterns
        repeat_score = calculate_repeat_score(component, grid_dimensions)
        pattern_score += max(symmetry_score, repeat_score)
    return min(pattern_score / len(components), 1) if components else 0

def calculate_symmetry_score(component: List[Tuple[int, int]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a symmetry score for a component."""
    rows, cols = grid_dimensions
    center_row, center_col = rows / 2, cols / 2
    symmetry_count = 0
    for x, y in component:
        if (2*center_row - x, y) in component:  # Horizontal symmetry
            symmetry_count += 1
        if (x, 2*center_col - y) in component:  # Vertical symmetry
            symmetry_count += 1
    return symmetry_count / (2 * len(component))

def calculate_repeat_score(component: List[Tuple[int, int]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score for repeating patterns in a component."""
    rows, cols = grid_dimensions
    repeat_count = 0
    for dx in range(1, cols // 2):
        for dy in range(1, rows // 2):
            if all((x+dx, y+dy) in component for x, y in component if x+dx < cols and y+dy < rows):
                repeat_count += 1
    return min(repeat_count / (rows * cols), 1)

def calculate_multi_scale_score(grid: ColoredGrid, color: int, grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score based on the presence of the color at multiple scales."""
    rows, cols = grid_dimensions
    scales = [1, 2, 4]  # Analyze at 1x1, 2x2, and 4x4 scales
    scale_scores = []
    
    for scale in scales:
        block_rows = rows // scale
        block_cols = cols // scale
        block_count = 0
        color_count = 0
        
        for i in range(0, rows, scale):
            for j in range(0, cols, scale):
                block = [grid.values[r][c] for r in range(i, min(i+scale, rows)) 
                                           for c in range(j, min(j+scale, cols))]
                block_count += 1
                if color in block:
                    color_count += 1
        
        scale_scores.append(color_count / block_count if block_count > 0 else 0)
    
    return sum(scale_scores) / len(scales)

def calculate_centrality_score(components: List[List[Tuple[int, int]]], grid_dimensions: Tuple[int, int]) -> float:
    """Calculate a score based on the centrality of the components."""
    rows, cols = grid_dimensions
    center_row, center_col = rows / 2, cols / 2
    max_distance = math.sqrt(center_row**2 + center_col**2)
    
    total_distance = 0
    total_cells = 0
    
    for component in components:
        for r, c in component:
            distance = math.sqrt((r - center_row)**2 + (c - center_col)**2)
            total_distance += distance
            total_cells += 1
    
    if total_cells == 0:
        return 0
    
    avg_distance = total_distance / total_cells
    centrality = 1 - (avg_distance / max_distance)
    return centrality

def calculate_relationship_score(grid: ColoredGrid, color: int, background_color: int) -> float:
    """Calculate a score based on the relationships between colors."""
    adjacent_colors = set()
    total_adjacencies = 0
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == color:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                        adjacent_color = grid.values[nr][nc]
                        if adjacent_color != color and adjacent_color != background_color:
                            adjacent_colors.add(adjacent_color)
                            total_adjacencies += 1
    
    if total_adjacencies == 0:
        return 0
    
    return len(adjacent_colors) / 9  # Normalize by total possible colors

def calculate_structural_score(grid: ColoredGrid, color: int, background_color: int) -> float:
    """Calculate a score based on the structural importance of the color."""
    rows, cols = grid.get_dimensions()
    total_cells = rows * cols
    color_cells = sum(row.count(color) for row in grid.values)
    
    # Create a copy of the grid with the color removed
    grid_without_color = ColoredGrid(values=[[cell if cell != color else background_color for cell in row] for row in grid.values])
    
    # Calculate the difference in connected components
    original_components = sum(len(grid.find_connected_regions(c)) for c in set(cell for row in grid.values for cell in row) if c != background_color)
    new_components = sum(len(grid_without_color.find_connected_regions(c)) for c in set(cell for row in grid_without_color.values for cell in row) if c != background_color)
    
    component_difference = abs(original_components - new_components)
    
    # Normalize the scores
    color_coverage = color_cells / total_cells
    component_impact = component_difference / original_components if original_components > 0 else 0
    
    # Combine the scores
    return (color_coverage + component_impact) / 2
