from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1a6449f1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a significant subgrid from the input grid based on the following steps:
    1. Analyze the input grid for color frequencies and patterns.
    2. Determine the output size based on the input dimensions.
    3. Define a search area in the bottom-right quadrant of the input grid.
    4. Generate candidate subgrids within the search area.
    5. Score each candidate based on color diversity, representation, and structural elements.
    6. Select and refine the best-scoring subgrid.
    7. Return the extracted subgrid as the output.
    """
    input_analysis = analyze_input_grid(input_grid)
    output_size = determine_output_size(input_grid.get_dimensions())
    search_area = define_search_area(input_grid, output_size)
    candidates = generate_candidates(input_grid, search_area, output_size)
    
    best_subgrid = max(candidates, key=lambda subgrid: score_subgrid(subgrid, input_analysis))
    refined_subgrid = refine_subgrid(input_grid, best_subgrid, search_area)
    
    return refined_subgrid

def analyze_input_grid(input_grid: ColoredGrid) -> Dict:
    rows, cols = input_grid.get_dimensions()
    color_frequencies = input_grid.get_color_frequencies()
    total_cells = rows * cols
    color_ratios = {color: count / total_cells for color, count in color_frequencies.items()}
    
    return {
        "dimensions": (rows, cols),
        "color_frequencies": color_frequencies,
        "color_ratios": color_ratios
    }

def determine_output_size(input_dimensions: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = input_dimensions
    output_rows = max(3, min(10, rows // 2))
    output_cols = max(3, min(10, cols // 2))
    return (output_rows, output_cols)

def define_search_area(input_grid: ColoredGrid, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    rows, cols = input_grid.get_dimensions()
    output_rows, output_cols = output_size
    
    start_row = max(0, rows - output_rows * 2)
    start_col = max(0, cols - output_cols * 2)
    
    return (start_row, start_col, rows, cols)

def generate_candidates(input_grid: ColoredGrid, search_area: Tuple[int, int, int, int], output_size: Tuple[int, int]) -> List[ColoredGrid]:
    start_row, start_col, end_row, end_col = search_area
    output_rows, output_cols = output_size
    
    candidates = []
    for row in range(start_row, end_row - output_rows + 1):
        for col in range(start_col, end_col - output_cols + 1):
            subgrid = input_grid.extract_subgrid(row, col, output_rows, output_cols)
            candidates.append(subgrid)
    
    return candidates

def score_subgrid(subgrid: ColoredGrid, input_analysis: Dict) -> float:
    subgrid_analysis = analyze_input_grid(subgrid)
    color_diversity = len(subgrid_analysis["color_frequencies"])
    color_representation = sum(min(subgrid_analysis["color_ratios"].get(color, 0), ratio) 
                               for color, ratio in input_analysis["color_ratios"].items())
    non_black_density = 1 - subgrid_analysis["color_ratios"].get(0, 0)
    
    return color_diversity * color_representation * non_black_density

def refine_subgrid(input_grid: ColoredGrid, selected_subgrid: ColoredGrid, search_area: Tuple[int, int, int, int]) -> ColoredGrid:
    start_row, start_col, end_row, end_col = search_area
    subgrid_rows, subgrid_cols = selected_subgrid.get_dimensions()
    
    best_subgrid = selected_subgrid
    best_score = score_subgrid(selected_subgrid, analyze_input_grid(input_grid))
    
    for row in range(start_row, end_row - subgrid_rows + 1):
        for col in range(start_col, end_col - subgrid_cols + 1):
            current_subgrid = input_grid.extract_subgrid(row, col, subgrid_rows, subgrid_cols)
            current_score = score_subgrid(current_subgrid, analyze_input_grid(input_grid))
            if current_score > best_score:
                best_score = current_score
                best_subgrid = current_subgrid
    
    return best_subgrid
