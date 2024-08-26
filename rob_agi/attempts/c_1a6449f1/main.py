from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1a6449f1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a significant subgrid from the input grid based on the following steps:
    1. Analyze the input grid for color frequencies, patterns, and density.
    2. Generate candidate regions of various sizes.
    3. Score candidates based on color diversity, pattern diversity, density, and uniqueness.
    4. Select and refine the best-scoring region.
    5. Adjust the output size if necessary.
    6. Post-process the output to remove empty edges while maintaining minimum size.
    7. Return the extracted subgrid as the output.
    """
    input_analysis = analyze_input_grid(input_grid)
    candidates = generate_candidates(input_grid, input_analysis)
    best_candidate = select_best_candidate(candidates, input_analysis)
    refined_subgrid = refine_subgrid(input_grid, best_candidate)
    output_subgrid = adjust_output_size(refined_subgrid)
    final_subgrid = post_process_output(output_subgrid)
    return final_subgrid

def analyze_input_grid(input_grid: ColoredGrid) -> Dict:
    rows, cols = input_grid.get_dimensions()
    color_frequencies = input_grid.get_color_frequencies()
    total_cells = rows * cols
    color_ratios = {color: count / total_cells for color, count in color_frequencies.items()}
    density_map = calculate_density_map(input_grid)
    patterns = identify_patterns(input_grid)
    
    return {
        "dimensions": (rows, cols),
        "color_frequencies": color_frequencies,
        "color_ratios": color_ratios,
        "density_map": density_map,
        "patterns": patterns
    }

def calculate_density_map(grid: ColoredGrid) -> List[List[float]]:
    rows, cols = grid.get_dimensions()
    density_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if 0 <= r + dr < rows and 0 <= c + dc < cols:
                            density_map[r + dr][c + dc] += 1
    
    max_density = max(max(row) for row in density_map)
    return [[cell / max_density for cell in row] for row in density_map]

def identify_patterns(grid: ColoredGrid) -> Dict[Tuple[int, ...], int]:
    rows, cols = grid.get_dimensions()
    patterns = defaultdict(int)
    
    for r in range(rows - 1):
        for c in range(cols - 1):
            pattern = (
                grid.get_cell(r, c),
                grid.get_cell(r, c + 1),
                grid.get_cell(r + 1, c),
                grid.get_cell(r + 1, c + 1)
            )
            patterns[pattern] += 1
    
    return dict(patterns)

def generate_candidates(input_grid: ColoredGrid, input_analysis: Dict) -> List[Tuple[ColoredGrid, Tuple[int, int]]]:
    rows, cols = input_grid.get_dimensions()
    candidates = []
    
    for size in range(3, 11):
        for r in range(rows - size + 1):
            for c in range(cols - size + 1):
                subgrid = input_grid.extract_subgrid(r, c, size, size)
                candidates.append((subgrid, (r, c)))
    
    return candidates

def score_candidate(candidate: ColoredGrid, input_analysis: Dict) -> float:
    color_diversity = len(set(cell for row in candidate.values for cell in row))
    pattern_diversity = len(identify_patterns(candidate))
    density = 1 - candidate.get_color_frequencies().get(0, 0) / (candidate.num_rows * candidate.num_cols)
    uniqueness = calculate_uniqueness(candidate, input_analysis)
    
    return (color_diversity * 0.3 + pattern_diversity * 0.3 + density * 0.2 + uniqueness * 0.2) / candidate.num_rows

def calculate_uniqueness(candidate: ColoredGrid, input_analysis: Dict) -> float:
    candidate_colors = set(cell for row in candidate.values for cell in row)
    input_colors = set(input_analysis["color_frequencies"].keys())
    return len(candidate_colors - {0}) / len(input_colors - {0})

def select_best_candidate(candidates: List[Tuple[ColoredGrid, Tuple[int, int]]], input_analysis: Dict) -> Tuple[ColoredGrid, Tuple[int, int]]:
    return max(candidates, key=lambda x: score_candidate(x[0], input_analysis))

def refine_subgrid(input_grid: ColoredGrid, best_candidate: Tuple[ColoredGrid, Tuple[int, int]]) -> ColoredGrid:
    subgrid, (r, c) = best_candidate
    rows, cols = input_grid.get_dimensions()
    size = subgrid.num_rows
    
    best_score = score_candidate(subgrid, analyze_input_grid(input_grid))
    best_subgrid = subgrid
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < rows - size + 1 and 0 <= new_c < cols - size + 1:
                new_subgrid = input_grid.extract_subgrid(new_r, new_c, size, size)
                new_score = score_candidate(new_subgrid, analyze_input_grid(input_grid))
                if new_score > best_score:
                    best_score = new_score
                    best_subgrid = new_subgrid
    
    return best_subgrid

def adjust_output_size(subgrid: ColoredGrid) -> ColoredGrid:
    if subgrid.num_rows > 10 or subgrid.num_cols > 10:
        return subgrid.extract_subgrid(0, 0, min(subgrid.num_rows, 10), min(subgrid.num_cols, 10))
    return subgrid

def post_process_output(subgrid: ColoredGrid) -> ColoredGrid:
    rows, cols = subgrid.get_dimensions()
    top, left, bottom, right = 0, 0, rows - 1, cols - 1
    
    while top < bottom and all(subgrid.get_cell(top, c) == 0 for c in range(cols)):
        top += 1
    while left < right and all(subgrid.get_cell(r, left) == 0 for r in range(rows)):
        left += 1
    while bottom > top and all(subgrid.get_cell(bottom, c) == 0 for c in range(cols)):
        bottom -= 1
    while right > left and all(subgrid.get_cell(r, right) == 0 for r in range(rows)):
        right -= 1
    
    height = max(3, bottom - top + 1)
    width = max(3, right - left + 1)
    
    return subgrid.extract_subgrid(top, left, height, width)
