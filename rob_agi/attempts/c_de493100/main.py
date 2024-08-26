from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_de493100(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the de493100 challenge by creating an abstracted version of the input grid.
    
    The solution works by:
    1. Analyzing input complexity and color frequencies
    2. Determining appropriate output size
    3. Creating a color importance map
    4. Analyzing color relationships and overall structure
    5. Creating an abstracted output grid based on the analysis
    6. Refining the output to balance color distribution and enhance contrast
    
    Args:
    input_grid (ColoredGrid): The input 30x30 grid

    Returns:
    ColoredGrid: An abstracted version of the input grid, sized between 4x4 and 10x10
    """
    # Step 1: Analyze Input Complexity
    color_freq = analyze_colors(input_grid)
    complexity = analyze_complexity(input_grid)
    
    # Step 2: Determine Output Size
    output_size = determine_output_size(complexity)
    
    # Step 3: Create Color Importance Map
    color_importance = create_color_importance_map(color_freq, input_grid)
    
    # Step 4: Analyze Color Relationships and Structure
    color_relationships = analyze_color_relationships(input_grid)
    overall_structure = analyze_overall_structure(input_grid)
    
    # Step 5: Create Initial Output Grid
    output_grid = create_initial_grid(output_size, color_importance, overall_structure)
    
    # Step 6: Refine Output Grid
    output_grid = refine_color_mapping(output_grid, input_grid, color_relationships)
    output_grid = balance_colors(output_grid, color_freq)
    output_grid = enhance_contrast(output_grid)
    
    return ColoredGrid(values=output_grid)

def analyze_colors(grid: ColoredGrid) -> Dict[int, int]:
    flat_grid = [color for row in grid.values for color in row]
    return dict(Counter(flat_grid))

def analyze_complexity(grid: ColoredGrid) -> int:
    unique_colors = len(set(color for row in grid.values for color in row))
    edge_count = sum(sum(1 for c in range(len(row)-1) if row[c] != row[c+1]) for row in grid.values)
    edge_count += sum(sum(1 for r in range(len(grid.values)-1) if grid.values[r][c] != grid.values[r+1][c]) for c in range(len(grid.values[0])))
    return unique_colors * 10 + edge_count

def determine_output_size(complexity: int) -> Tuple[int, int]:
    size = min(max(4, complexity // 40), 10)
    return (size, size)

def create_color_importance_map(color_freq: Dict[int, int], grid: ColoredGrid) -> List[int]:
    total_pixels = sum(color_freq.values())
    color_importance = sorted(color_freq.keys(), key=lambda c: color_freq[c] / total_pixels, reverse=True)
    return color_importance[:min(len(color_importance), 10)]

def analyze_color_relationships(grid: ColoredGrid) -> Dict[Tuple[int, int], int]:
    relationships = {}
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            color = grid.values[r][c]
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                    neighbor_color = grid.values[nr][nc]
                    if color != neighbor_color:
                        pair = tuple(sorted([color, neighbor_color]))
                        relationships[pair] = relationships.get(pair, 0) + 1
    return relationships

def analyze_overall_structure(grid: ColoredGrid) -> Dict[str, List[float]]:
    rows, cols = len(grid.values), len(grid.values[0])
    color_positions = {color: [] for color in range(10)}
    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            color_positions[color].append((r / rows, c / cols))
    
    structure = {}
    for color, positions in color_positions.items():
        if positions:
            avg_r = sum(p[0] for p in positions) / len(positions)
            avg_c = sum(p[1] for p in positions) / len(positions)
            structure[color] = [avg_r, avg_c]
    return structure

def create_initial_grid(output_size: Tuple[int, int], color_importance: List[int], overall_structure: Dict[str, List[float]]) -> List[List[int]]:
    grid = [[0 for _ in range(output_size[1])] for _ in range(output_size[0])]
    for color in color_importance:
        if color in overall_structure:
            r, c = overall_structure[color]
            grid_r = int(r * output_size[0])
            grid_c = int(c * output_size[1])
            grid[grid_r][grid_c] = color
    return grid

def refine_color_mapping(output_grid: List[List[int]], input_grid: ColoredGrid, color_relationships: Dict[Tuple[int, int], int]) -> List[List[int]]:
    for r in range(len(output_grid)):
        for c in range(len(output_grid[0])):
            if output_grid[r][c] == 0:
                best_color = max(range(10), key=lambda color: sum(
                    color_relationships.get((color, output_grid[nr][nc]), 0)
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    if 0 <= nr < len(output_grid) and 0 <= nc < len(output_grid[0])
                ))
                output_grid[r][c] = best_color
    return output_grid

def balance_colors(grid: List[List[int]], color_freq: Dict[int, int]) -> List[List[int]]:
    total_pixels = sum(color_freq.values())
    target_freq = {color: count / total_pixels for color, count in color_freq.items()}
    current_freq = Counter(color for row in grid for color in row)
    total_output_pixels = len(grid) * len(grid[0])
    current_freq = {color: count / total_output_pixels for color, count in current_freq.items()}
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            current_color = grid[r][c]
            if current_freq[current_color] > target_freq.get(current_color, 0):
                new_color = min(color_freq.keys(), key=lambda x: current_freq.get(x, 0) - target_freq.get(x, 0))
                grid[r][c] = new_color
                current_freq[current_color] -= 1 / total_output_pixels
                current_freq[new_color] = current_freq.get(new_color, 0) + 1 / total_output_pixels
    
    return grid

def enhance_contrast(grid: List[List[int]]) -> List[List[int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            neighbors = [
                grid[nr][nc]
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0])
            ]
            if all(grid[r][c] == neighbor for neighbor in neighbors):
                grid[r][c] = (grid[r][c] + 1) % 10
    return grid
