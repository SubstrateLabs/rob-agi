from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_67b4a34d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by analyzing the 16x16 input grid and creating a new 4x4 output grid.
    
    The solution analyzes the global structure of the input grid, including corners, edges,
    and center. It creates a color importance map based on frequency and structural significance.
    The output grid is generated to reflect key patterns, maintain symmetry, and ensure color diversity.
    
    Args:
    input_grid (ColoredGrid): A 16x16 input grid
    
    Returns:
    ColoredGrid: A 4x4 grid that captures the essence of the input grid's structure and color distribution
    """
    def analyze_region(region):
        flat = [cell for row in region for cell in row]
        return Counter(flat).most_common()

    # Analyze corners
    corners = [
        input_grid.extract_subgrid(0, 0, 4, 4),
        input_grid.extract_subgrid(0, 12, 4, 4),
        input_grid.extract_subgrid(12, 0, 4, 4),
        input_grid.extract_subgrid(12, 12, 4, 4)
    ]
    corner_colors = [analyze_region(c.values)[0][0] for c in corners]

    # Analyze edges
    edges = [
        [row[:4] + row[-4:] for row in input_grid.values[:4] + input_grid.values[-4:]],
        [row[4:12] for row in input_grid.values[:4] + input_grid.values[-4:]]
    ]
    edge_colors = [color for edge in edges for color, _ in analyze_region(edge)[:2]]

    # Analyze center
    center = input_grid.extract_subgrid(4, 4, 8, 8)
    center_colors = [color for color, _ in analyze_region(center.values)[:3]]

    # Create color importance map
    color_importance = Counter(corner_colors + edge_colors + center_colors)

    # Generate output grid
    output = [[0 for _ in range(4)] for _ in range(4)]

    # Set corners
    output[0][0] = output[3][3] = corner_colors[0]
    output[0][3] = output[3][0] = corner_colors[1]

    # Set edges
    output[0][1] = output[0][2] = edge_colors[0]
    output[1][0] = output[2][0] = edge_colors[1]
    output[3][1] = output[3][2] = edge_colors[2]
    output[1][3] = output[2][3] = edge_colors[3]

    # Set center
    output[1][1] = output[2][2] = center_colors[0]
    output[1][2] = output[2][1] = center_colors[1]

    # Ensure color diversity
    unique_colors = set(cell for row in output for cell in row)
    if len(unique_colors) < 3:
        additional_color = max(color_importance, key=lambda x: color_importance[x] if x not in unique_colors else 0)
        output[1][1] = output[2][2] = additional_color

    return ColoredGrid(values=output)
