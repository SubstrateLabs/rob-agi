from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1a6449f1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a representative signature subgrid from the input grid based on the following steps:
    1. Analyze the input grid for color frequencies, patterns, and uniqueness.
    2. Generate candidate regions of various sizes (3x3 to 10x10).
    3. Create signature subgrids for each candidate, compressing larger regions if necessary.
    4. Score signature subgrids based on representativeness, diversity, uniqueness, and compactness.
    5. Select the best-scoring signature subgrid.
    6. Refine the selected subgrid to better capture the input grid's essence.
    7. Post-process the output to ensure size constraints and remove empty edges.
    8. Return the final representative subgrid as the output.
    """
    input_analysis = analyze_input_grid(input_grid)
    candidates = generate_candidates(input_grid, input_analysis)
    signature_subgrids = create_signature_subgrids(candidates, input_analysis)
    best_signature = select_best_signature(signature_subgrids, input_analysis)
    refined_subgrid = refine_signature(input_grid, best_signature, input_analysis)
    final_subgrid = post_process_output(refined_subgrid)
    return final_subgrid

def analyze_input_grid(input_grid: ColoredGrid) -> Dict:
    rows, cols = input_grid.get_dimensions()
    color_frequencies = input_grid.get_color_frequencies()
    total_cells = rows * cols
    color_ratios = {color: count / total_cells for color, count in color_frequencies.items()}
    uniqueness_map = calculate_uniqueness_map(input_grid, color_frequencies)
    patterns = identify_patterns(input_grid)
    
    return {
        "dimensions": (rows, cols),
        "color_frequencies": color_frequencies,
        "color_ratios": color_ratios,
        "uniqueness_map": uniqueness_map,
        "patterns": patterns
    }

def calculate_uniqueness_map(grid: ColoredGrid, color_frequencies: Dict[int, int]) -> List[List[float]]:
    rows, cols = grid.get_dimensions()
    total_cells = rows * cols
    uniqueness_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            uniqueness = 1 - (color_frequencies[color] / total_cells)
            uniqueness_map[r][c] = uniqueness
    
    return uniqueness_map

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

def create_signature_subgrids(candidates: List[Tuple[ColoredGrid, Tuple[int, int]]], input_analysis: Dict) -> List[ColoredGrid]:
    signatures = []
    for subgrid, _ in candidates:
        if subgrid.num_rows <= 5 and subgrid.num_cols <= 5:
            signatures.append(subgrid)
        else:
            compressed = compress_subgrid(subgrid)
            signatures.append(compressed)
    return signatures

def compress_subgrid(subgrid: ColoredGrid) -> ColoredGrid:
    rows, cols = subgrid.get_dimensions()
    if rows <= 5 and cols <= 5:
        return subgrid
    
    new_rows = min(rows, 5)
    new_cols = min(cols, 5)
    compressed = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    for r in range(new_rows):
        for c in range(new_cols):
            r_start, r_end = r * rows // new_rows, (r + 1) * rows // new_rows
            c_start, c_end = c * cols // new_cols, (c + 1) * cols // new_cols
            colors = [subgrid.get_cell(rr, cc) for rr in range(r_start, r_end) for cc in range(c_start, c_end)]
            compressed[r][c] = max(set(colors), key=colors.count)
    
    return ColoredGrid(values=compressed)

def score_signature(signature: ColoredGrid, input_analysis: Dict) -> float:
    color_diversity = len(set(cell for row in signature.values for cell in row) - {0})
    pattern_diversity = len(identify_patterns(signature))
    uniqueness = calculate_uniqueness(signature, input_analysis)
    compactness = 1 / (signature.num_rows * signature.num_cols)
    representativeness = calculate_representativeness(signature, input_analysis)
    
    return (color_diversity * 0.2 + pattern_diversity * 0.2 + uniqueness * 0.2 + 
            compactness * 0.2 + representativeness * 0.2)

def calculate_uniqueness(signature: ColoredGrid, input_analysis: Dict) -> float:
    signature_colors = set(cell for row in signature.values for cell in row) - {0}
    input_colors = set(input_analysis["color_frequencies"].keys()) - {0}
    return len(signature_colors) / len(input_colors) if input_colors else 0

def calculate_representativeness(signature: ColoredGrid, input_analysis: Dict) -> float:
    signature_ratios = signature.get_color_frequencies()
    input_ratios = input_analysis["color_ratios"]
    total_diff = sum(abs(signature_ratios.get(color, 0) - ratio) for color, ratio in input_ratios.items())
    return 1 - (total_diff / 2)  # Normalize to [0, 1]

def select_best_signature(signatures: List[ColoredGrid], input_analysis: Dict) -> ColoredGrid:
    return max(signatures, key=lambda x: score_signature(x, input_analysis))

def refine_signature(input_grid: ColoredGrid, signature: ColoredGrid, input_analysis: Dict) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    sig_rows, sig_cols = signature.get_dimensions()
    best_score = score_signature(signature, input_analysis)
    best_subgrid = signature
    
    for r in range(rows - sig_rows + 1):
        for c in range(cols - sig_cols + 1):
            subgrid = input_grid.extract_subgrid(r, c, sig_rows, sig_cols)
            score = score_signature(subgrid, input_analysis)
            if score > best_score:
                best_score = score
                best_subgrid = subgrid
    
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
    
    height = max(3, min(bottom - top + 1, 10))
    width = max(3, min(right - left + 1, 10))
    
    processed = subgrid.extract_subgrid(top, left, height, width)
    
    if processed.num_rows < 3 or processed.num_cols < 3:
        return expand_subgrid(processed)
    elif processed.num_rows > 10 or processed.num_cols > 10:
        return compress_subgrid(processed)
    else:
        return processed

def expand_subgrid(subgrid: ColoredGrid) -> ColoredGrid:
    rows, cols = subgrid.get_dimensions()
    new_rows = max(3, rows)
    new_cols = max(3, cols)
    new_values = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    for r in range(rows):
        for c in range(cols):
            new_values[r][c] = subgrid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)
