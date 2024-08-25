from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by finding the most representative repeating pattern in the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to identify non-black areas.
    2. Identify potential pattern sizes.
    3. For each potential size, find the pattern with the highest coverage of the non-black area.
    4. Select the best pattern based on coverage and size.
    5. Refine the pattern to ensure it's complete and doesn't include unnecessary black cells.
    6. Extract the final subgrid from the original input.
    7. Validate the output and handle edge cases.
    
    Returns:
        ColoredGrid: A new grid containing the extracted pattern.
    """
    rows, cols = input_grid.get_dimensions()
    non_black_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) != 0]
    
    if not non_black_cells:
        return ColoredGrid(values=[[0, 0], [0, 0], [0, 0]])
    
    def get_pattern_coverage(pattern: List[List[int]]) -> float:
        pattern_rows, pattern_cols = len(pattern), len(pattern[0])
        covered_cells = 0
        total_cells = len(non_black_cells)
        
        for r, c in non_black_cells:
            if input_grid.get_cell(r, c) == pattern[r % pattern_rows][c % pattern_cols]:
                covered_cells += 1
        
        return covered_cells / total_cells
    
    best_pattern = None
    best_coverage = 0
    
    for height in range(2, min(9, rows + 1)):
        for width in range(2, min(9, cols + 1)):
            if height * width > 40:  # Max size based on examples
                continue
            
            for r in range(rows - height + 1):
                for c in range(cols - width + 1):
                    pattern = [
                        [input_grid.get_cell(r + i, c + j) for j in range(width)]
                        for i in range(height)
                    ]
                    coverage = get_pattern_coverage(pattern)
                    
                    if coverage > best_coverage or (coverage == best_coverage and height * width < len(best_pattern) * len(best_pattern[0])):
                        best_pattern = pattern
                        best_coverage = coverage
    
    if best_pattern is None:
        # Fallback: return the largest non-black rectangular area up to 8x5
        non_black_rows = [r for r in range(rows) if any(input_grid.get_cell(r, c) != 0 for c in range(cols))]
        non_black_cols = [c for c in range(cols) if any(input_grid.get_cell(r, c) != 0 for r in range(rows))]
        
        height = min(8, len(non_black_rows))
        width = min(5, len(non_black_cols))
        
        return ColoredGrid(values=[
            [input_grid.get_cell(non_black_rows[r], non_black_cols[c]) for c in range(width)]
            for r in range(height)
        ])
    
    # Refine the pattern by removing unnecessary black cells
    while any(all(cell == 0 for cell in row) for row in best_pattern):
        best_pattern = [row for row in best_pattern if any(cell != 0 for cell in row)]
    while any(all(best_pattern[r][c] == 0 for r in range(len(best_pattern))) for c in range(len(best_pattern[0]))):
        best_pattern = [[row[c] for c in range(len(best_pattern[0])) if any(best_pattern[r][c] != 0 for r in range(len(best_pattern)))] for row in best_pattern]
    
    return ColoredGrid(values=best_pattern)
