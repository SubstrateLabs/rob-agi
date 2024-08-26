from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending an existing gray 'L' shape.
    
    The solution follows these steps:
    1. Identify the existing gray 'L' shape.
    2. Extend the 'L' shape downward to near the bottom of the grid.
    3. Extend the path rightward to about 1/2 to 2/3 of the grid width.
    4. Optionally add a small upward extension if space allows.
    5. Update the grid with the new path of 5's (gray).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with an extended gray path.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_existing_l() -> Tuple[int, int, int, int]:
        # Find vertical part of L
        vert_len = 0
        for r in range(rows):
            if output_grid.get_cell(r, 0) != 5:
                break
            vert_len += 1
        
        # Find horizontal part of L
        horz_len = 0
        for c in range(cols):
            if output_grid.get_cell(vert_len-1, c) != 5:
                break
            horz_len += 1
        
        return 0, 0, vert_len, horz_len
    
    def extend_path(start_r: int, start_c: int) -> List[Tuple[int, int]]:
        path = []
        r, c = start_r, start_c
        
        # Extend downward
        target_row = min(rows - 2, rows - 1)
        while r < target_row and output_grid.get_cell(r, c) != 5:
            path.append((r, c))
            r += 1
        
        # Extend rightward
        target_col = min(cols - 1, cols * 2 // 3)
        while c < target_col and output_grid.get_cell(r, c) != 5:
            path.append((r, c))
            c += 1
        
        # Optional upward extension
        if r > 3 and len(path) < 15:
            for _ in range(min(3, r - 3)):
                r -= 1
                path.append((r, c))
        
        return path
    
    # Find existing L
    start_r, start_c, vert_len, horz_len = find_existing_l()
    
    # Extend the path
    extension = extend_path(vert_len-1, horz_len-1)
    
    # Update grid with new path
    for r, c in extension:
        output_grid.set_cell(r, c, 5)
    
    return output_grid
