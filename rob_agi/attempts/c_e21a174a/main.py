from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e21a174a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the e21a174a challenge by rearranging color groups vertically within sections.
    
    The function identifies full-width horizontal lines as section separators,
    then within each section, it rearranges connected color groups from bottom to top
    based on their lowest point. The internal structure and horizontal position of each shape
    is preserved, and the overall vertical order of sections is maintained.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with color groups rearranged within sections.
    """
    rows, cols = input_grid.get_dimensions()
    
    def get_connected_group(start_row: int, start_col: int, color: int, section_start: int, section_end: int) -> List[Tuple[int, int]]:
        group = []
        stack = [(start_row, start_col)]
        visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and section_start <= r <= section_end and 0 <= c < cols and input_grid.values[r][c] == color:
                visited.add((r, c))
                group.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        
        return group

    def is_separator(row: List[int]) -> bool:
        return len(set(row[1:-1])) == 1 and row[1] != 0

    # Step 1: Identify section separators
    separators = [i for i, row in enumerate(input_grid.values) if is_separator(row)]
    separators = [-1] + separators + [rows]

    # Step 2: Process each section
    output_values = [row[:] for row in input_grid.values]
    for i in range(len(separators) - 1):
        section_start = separators[i] + 1
        section_end = separators[i + 1] - 1
        
        if section_start > section_end:
            continue  # Skip empty sections
        
        # Identify connected color groups within the section
        color_groups = []
        visited = set()
        for row in range(section_start, section_end + 1):
            for col in range(cols):
                if (row, col) not in visited and input_grid.values[row][col] != 0:
                    group = get_connected_group(row, col, input_grid.values[row][col], section_start, section_end)
                    color_groups.append({
                        'color': input_grid.values[row][col],
                        'cells': group,
                        'bottom': max(r for r, _ in group)
                    })
                    visited.update(group)
        
        # Sort color groups from bottom to top
        color_groups.sort(key=lambda g: g['bottom'], reverse=True)
        
        # Rearrange shapes within the section
        new_section = [[0 for _ in range(cols)] for _ in range(section_end - section_start + 1)]
        current_row = section_end - section_start
        for group in color_groups:
            group_height = max(r for r, _ in group['cells']) - min(r for r, _ in group['cells']) + 1
            new_bottom = current_row
            shift = new_bottom - (group['bottom'] - section_start)
            
            for old_row, col in group['cells']:
                new_row = old_row - section_start + shift
                new_section[new_row][col] = group['color']
            
            current_row = new_bottom - group_height
        
        # Update the output grid with the rearranged section
        for row in range(section_start, section_end + 1):
            output_values[row] = new_section[row - section_start]

    # Step 3: Return the new ColoredGrid
    return ColoredGrid(values=output_values)
