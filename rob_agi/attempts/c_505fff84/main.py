from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_505fff84(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern of red squares from the input grid.
    
    The function performs the following steps:
    1. Identifies connected regions of red squares
    2. Extracts and simplifies subgrids for each significant region
    3. Scores potential patterns based on compactness and simplicity
    4. Selects and refines the best pattern
    5. Returns the final pattern as a new ColoredGrid
    """
    # Step 1: Identify connected regions of red squares
    red_regions = input_grid.find_connected_regions(2)
    
    # Step 2 & 3: Extract, simplify, and score potential patterns
    patterns = []
    for region in red_regions:
        if len(region) < 4:  # Ignore very small regions
            continue
        subgrid = extract_subgrid(input_grid, region)
        simplified = simplify_subgrid(subgrid)
        score = score_pattern(simplified)
        patterns.append((simplified, score))
    
    # Step 4: Select the best pattern
    if not patterns:
        return ColoredGrid(values=[[0]])  # Return a 1x1 black grid if no patterns found
    best_pattern, _ = max(patterns, key=lambda x: x[1])
    
    # Step 5: Refine the pattern
    final_pattern = refine_pattern(best_pattern)
    
    return final_pattern

def extract_subgrid(grid: ColoredGrid, region: List[Tuple[int, int]]) -> ColoredGrid:
    top = min(r for r, _ in region)
    left = min(c for _, c in region)
    bottom = max(r for r, _ in region)
    right = max(c for _, c in region)
    return grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)

def simplify_subgrid(grid: ColoredGrid) -> ColoredGrid:
    new_values = [[2 if cell == 2 else 0 for cell in row] for row in grid.values]
    return ColoredGrid(values=new_values)

def score_pattern(grid: ColoredGrid) -> float:
    total_cells = sum(len(row) for row in grid.values)
    red_cells = sum(row.count(2) for row in grid.values)
    compactness = red_cells / total_cells
    simplicity = 1 / (grid.num_rows + grid.num_cols)  # Prefer smaller grids
    return compactness * simplicity

def refine_pattern(grid: ColoredGrid) -> ColoredGrid:
    # Remove empty rows and columns
    rows_to_keep = [i for i, row in enumerate(grid.values) if any(cell == 2 for cell in row)]
    cols_to_keep = [j for j in range(grid.num_cols) if any(row[j] == 2 for row in grid.values)]
    
    new_values = [[grid.values[i][j] for j in cols_to_keep] for i in rows_to_keep]
    return ColoredGrid(values=new_values)
