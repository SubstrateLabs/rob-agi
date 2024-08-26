from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red diamond shapes with blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) diamonds
    2. Group red diamonds horizontally
    3. Connect diamonds within each horizontal group
    4. Identify and create vertical connections between groups
    5. Optimize the solution by removing unnecessary lines
    
    This approach creates a minimal structure of blue lines that connects all red diamonds,
    focusing on horizontal connections within groups and vertical connections between groups.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify red diamonds
    red_diamonds = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2]
    
    if not red_diamonds:
        return output_grid
    
    # Step 2: Group red diamonds horizontally
    red_diamonds.sort()  # Sort by row, then by column
    groups = []
    current_group = [red_diamonds[0]]
    for diamond in red_diamonds[1:]:
        if diamond[0] - current_group[-1][0] <= 1:  # Same or adjacent row
            current_group.append(diamond)
        else:
            groups.append(current_group)
            current_group = [diamond]
    groups.append(current_group)
    
    # Step 3: Connect diamonds within each horizontal group
    for group in groups:
        min_row = min(d[0] for d in group)
        max_row = max(d[0] for d in group)
        min_col = min(d[1] for d in group)
        max_col = max(d[1] for d in group)
        
        for r in range(min_row, max_row + 1):
            diamonds_in_row = [d for d in group if d[0] == r]
            if diamonds_in_row:
                left = min(d[1] for d in diamonds_in_row)
                right = max(d[1] for d in diamonds_in_row)
                for c in range(left, right + 1):
                    if output_grid.values[r][c] != 2:
                        output_grid.values[r][c] = 1
    
    # Step 4: Identify and create vertical connections
    columns_to_connect = set()
    for c in range(cols):
        groups_in_column = [g for g in groups if any(d[1] == c for d in g)]
        if len(groups_in_column) > 1:
            columns_to_connect.add(c)
    
    for c in columns_to_connect:
        diamonds_in_column = [d for d in red_diamonds if d[1] == c]
        top = min(d[0] for d in diamonds_in_column)
        bottom = max(d[0] for d in diamonds_in_column)
        for r in range(top, bottom + 1):
            if output_grid.values[r][c] != 2:
                output_grid.values[r][c] = 1
    
    # Step 5: Optimize by removing unnecessary lines
    # (This step is not implemented here as it requires more complex logic)
    
    return output_grid
