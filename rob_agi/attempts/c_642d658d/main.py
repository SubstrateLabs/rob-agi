from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import math

def solve_642d658d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 642d658d challenge by identifying the most significant color in the input grid.
    
    The solution follows these steps:
    1. Analyze the input grid to determine background color and non-background colors
    2. For each non-background color, calculate a comprehensive score based on:
       - Frequency
       - Structural importance (connected components)
       - Pattern formation (lines, symmetry, shapes)
       - Color interactions
       - Distribution across the grid
       - Positional importance
       - Multi-scale presence
    3. Combine the scores using weighted sum
    4. Select the color with the highest combined score
    5. Return a 1x1 grid with the selected color
    
    This approach considers multiple aspects of color significance, allowing for a
    comprehensive analysis of the input grid to determine the most important color.
    """
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    background_color = max(color_counts, key=color_counts.get)
    total_cells = sum(color_counts.values())
    
    non_background_colors = set(color_counts.keys()) - {background_color}
    
    if not non_background_colors:
        return ColoredGrid(values=[[background_color]])
    
    color_scores = {}
    for color in non_background_colors:
        frequency_score = color_counts[color] / total_cells
        structural_score = calculate_structural_score(input_grid, color)
        pattern_score = calculate_pattern_score(input_grid, color)
        interaction_score = calculate_interaction_score(input_grid, color, background_color)
        distribution_score = calculate_distribution_score(input_grid, color)
        position_score = calculate_position_score(input_grid, color)
        multi_scale_score = calculate_multi_scale_score(input_grid, color)
        
        # Combine scores with weights
        color_scores[color] = (
            frequency_score * 0.15 +
            structural_score * 0.2 +
            pattern_score * 0.15 +
            interaction_score * 0.15 +
            distribution_score * 0.1 +
            position_score * 0.15 +
            multi_scale_score * 0.1
        )
    
    selected_color = max(color_scores, key=color_scores.get)
    return ColoredGrid(values=[[selected_color]])

def find_connected_components(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected components of a given color in the grid."""
    return grid.find_connected_regions(color)

def calculate_structural_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the structural importance score for a color."""
    components = grid.find_connected_regions(color)
    if not components:
        return 0
    
    avg_size = sum(len(comp) for comp in components) / len(components)
    max_size = max(len(comp) for comp in components)
    
    return (avg_size + max_size) / (2 * grid.num_rows * grid.num_cols)

def calculate_pattern_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the pattern formation score for a color."""
    rows, cols = grid.get_dimensions()
    horizontal_lines = sum(1 for row in grid.values if any(cell == color for cell in row))
    vertical_lines = sum(1 for c in range(cols) if any(grid.values[r][c] == color for r in range(rows)))
    diagonal_lines = sum(1 for i in range(rows + cols - 1) if any(grid.values[r][c] == color for r, c in zip(range(rows), range(i, -1, -1)) if c < cols))
    
    symmetry_score = calculate_symmetry_score(grid, color)
    
    return (horizontal_lines / rows + vertical_lines / cols + diagonal_lines / (rows + cols) + symmetry_score) / 4

def calculate_symmetry_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the symmetry score for a color."""
    rows, cols = grid.get_dimensions()
    horizontal_symmetry = sum(grid.values[r] == grid.values[rows - 1 - r] for r in range(rows // 2)) / (rows // 2)
    vertical_symmetry = sum(all(grid.values[r][c] == grid.values[r][cols - 1 - c] for r in range(rows)) for c in range(cols // 2)) / (cols // 2)
    return (horizontal_symmetry + vertical_symmetry) / 2

def calculate_interaction_score(grid: ColoredGrid, color: int, background_color: int) -> float:
    """Calculate the color interaction score."""
    rows, cols = grid.get_dimensions()
    interactions = 0
    color_cells = 0
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == color:
                color_cells += 1
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] not in [color, background_color]:
                        interactions += 1
    return interactions / (4 * color_cells) if color_cells > 0 else 0

def calculate_distribution_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the distribution score for a color."""
    rows, cols = grid.get_dimensions()
    sector_size = max(rows, cols) // 3
    sectors = set()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == color:
                sectors.add((r // sector_size, c // sector_size))
    return len(sectors) / 9  # 9 is the maximum number of sectors

def calculate_position_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the positional importance score for a color."""
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    total_distance = 0
    color_cells = 0
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == color:
                color_cells += 1
                distance = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                total_distance += 1 - (distance / max(center_r, center_c))
    return total_distance / color_cells if color_cells > 0 else 0

def calculate_multi_scale_score(grid: ColoredGrid, color: int) -> float:
    """Calculate the multi-scale presence score for a color."""
    rows, cols = grid.get_dimensions()
    scales = [1, 2, 4]
    scale_scores = []
    
    for scale in scales:
        block_rows = rows // scale
        block_cols = cols // scale
        blocks_with_color = 0
        total_blocks = 0
        
        for r in range(0, rows, scale):
            for c in range(0, cols, scale):
                total_blocks += 1
                if any(grid.values[rr][cc] == color 
                       for rr in range(r, min(r + scale, rows)) 
                       for cc in range(c, min(c + scale, cols))):
                    blocks_with_color += 1
        
        scale_scores.append(blocks_with_color / total_blocks if total_blocks > 0 else 0)
    
    return sum(scale_scores) / len(scales)

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
