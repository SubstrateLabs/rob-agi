from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_47996f11(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by removing the magenta region and extending surrounding patterns.
    
    The solution involves:
    1. Identifying the magenta (6) region in each column
    2. Analyzing patterns above and below the magenta area
    3. Generating new sequences to replace magenta, based on surrounding patterns
    4. Preserving horizontal patterns and smoothing transitions
    5. Handling edge cases where magenta touches grid boundaries
    6. Performing a final smoothing pass to ensure pattern continuity
    
    This approach maintains the overall color distribution and extends existing patterns
    through the area previously occupied by magenta, while ensuring smooth transitions
    and preserving the integrity of surrounding patterns.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def find_magenta_boundaries(col: int) -> Tuple[int, int]:
        top = next((r for r in range(rows) if output_grid.values[r][col] == 6), -1)
        bottom = next((r for r in range(rows-1, -1, -1) if output_grid.values[r][col] == 6), -1)
        return top, bottom
    
    def analyze_pattern(col: int, start: int, end: int, direction: int) -> List[int]:
        return [output_grid.values[r][col] for r in range(start, end, direction) if output_grid.values[r][col] != 6]
    
    def generate_replacement(top_pattern: List[int], bottom_pattern: List[int], height: int) -> List[int]:
        if not top_pattern and not bottom_pattern:
            return [0] * height  # Default to black if no patterns available
        elif not top_pattern:
            return (bottom_pattern * (height // len(bottom_pattern) + 1))[:height]
        elif not bottom_pattern:
            return (top_pattern * (height // len(top_pattern) + 1))[:height]
        
        third = height // 3
        top_part = (top_pattern * (third // len(top_pattern) + 1))[:third]
        bottom_part = (bottom_pattern * (third // len(bottom_pattern) + 1))[:third]
        middle_part = [((top_pattern[i % len(top_pattern)] + bottom_pattern[i % len(bottom_pattern)]) // 2) 
                       for i in range(height - 2*third)]
        return top_part + middle_part + bottom_part
    
    for col in range(cols):
        top, bottom = find_magenta_boundaries(col)
        if top == -1 or bottom == -1:
            continue  # No magenta in this column
        
        top_pattern = analyze_pattern(col, 0, top, 1)
        bottom_pattern = analyze_pattern(col, rows-1, bottom, -1)
        replacement = generate_replacement(top_pattern, bottom_pattern, bottom - top + 1)
        
        for i, r in enumerate(range(top, bottom + 1)):
            output_grid.values[r][col] = replacement[i]
    
    # Horizontal smoothing pass
    for r in range(rows):
        for c in range(1, cols-1):
            if input_grid.values[r][c] == 6:
                neighbors = [output_grid.values[r][c-1], output_grid.values[r][c+1]]
                output_grid.values[r][c] = max(set(neighbors), key=neighbors.count)
    
    # Final smoothing pass
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if input_grid.values[r][c] == 6:
                neighbors = [output_grid.values[r+dr][c+dc] for dr in [-1,0,1] for dc in [-1,0,1] if (dr,dc) != (0,0)]
                output_grid.values[r][c] = max(set(neighbors), key=neighbors.count)
    
    return output_grid
