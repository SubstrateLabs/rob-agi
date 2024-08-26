from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending an existing gray 'L' shape.
    
    The solution follows these steps:
    1. Identify the existing gray 'L' shape in the top-left corner.
    2. Extend the path downward and rightward, aiming to add about 1/3 to 1/2 of the grid size.
    3. Add vertical segments to ensure connectivity and create complex patterns.
    4. Update the grid with the new path of 5's (gray).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with an extended gray path.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_existing_l() -> Tuple[int, int, int, int]:
        vert_len = 0
        for r in range(rows):
            if output_grid.get_cell(r, 0) != 5:
                break
            vert_len += 1
        
        horz_len = 0
        for c in range(cols):
            if output_grid.get_cell(vert_len-1, c) != 5:
                break
            horz_len += 1
        
        return 0, 0, vert_len, horz_len
    
    def extend_path(start_r: int, start_c: int) -> List[Tuple[int, int]]:
        path = []
        r, c = start_r, start_c
        target_cells = (rows * cols) // 3  # Aim to add about 1/3 of the grid size
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        dir_index = 0
        
        while len(path) < target_cells:
            dr, dc = directions[dir_index]
            new_r, new_c = r + dr, c + dc
            
            if 0 <= new_r < rows and 0 <= new_c < cols and output_grid.get_cell(new_r, new_c) != 5:
                path.append((new_r, new_c))
                r, c = new_r, new_c
            else:
                dir_index = (dir_index + 1) % 4  # Try next direction
                
            if dir_index == 0:  # If we've tried all directions, break
                break
        
        return path
    
    def add_vertical_segments(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        columns = set(c for _, c in path)
        for c in columns:
            column_cells = sorted([r for r, col in path if col == c])
            for i in range(len(column_cells) - 1):
                for r in range(column_cells[i] + 1, column_cells[i+1]):
                    path.append((r, c))
        return path
    
    # Find existing L
    start_r, start_c, vert_len, horz_len = find_existing_l()
    
    # Extend the path
    extension = extend_path(vert_len-1, horz_len-1)
    
    # Add vertical segments
    extension = add_vertical_segments(extension)
    
    # Update grid with new path
    for r, c in extension:
        output_grid.set_cell(r, c, 5)
    
    return output_grid
