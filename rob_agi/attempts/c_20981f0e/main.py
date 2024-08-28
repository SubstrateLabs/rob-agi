from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_20981f0e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Rearrange blue cells (1) in each section between red dot (2) rows to form vertically aligned columns.
    The solution maintains the same number of blue cells in each section, preserves the positions of red dots,
    and aligns blue cells both vertically and horizontally across all sections.
    Steps:
    1. Analyze the input grid and identify sections between red dot rows
    2. Determine the leftmost and rightmost columns for blue cells in each section
    3. Calculate the number of blue cells for left and right columns in each section
    4. Rearrange blue cells in each section, ensuring vertical alignment and proper spacing
    5. Construct the output grid with the new arrangements
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find red dot rows and copy them to output grid
    red_dot_rows = []
    for i in range(rows):
        if 2 in input_grid.values[i]:
            red_dot_rows.append(i)
            output_grid.values[i] = input_grid.values[i].copy()
    
    # Define sections
    sections = list(zip([-1] + red_dot_rows, red_dot_rows + [rows]))
    
    # Analyze all sections
    section_info = analyze_sections(input_grid, sections)
    
    # Rearrange blue cells in each section
    for (start, end), info in zip(sections, section_info):
        if end - start > 2:
            rearrange_section(output_grid, start, end, info)
    
    return output_grid

def analyze_sections(input_grid: ColoredGrid, sections: List[Tuple[int, int]]) -> List[dict]:
    section_info = []
    max_left_count = 0
    
    for start, end in sections:
        if end - start <= 2:
            section_info.append(None)
            continue
        
        blue_count = sum(input_grid.values[r][c] == 1 
                         for r in range(start + 1, end) 
                         for c in range(input_grid.get_dimensions()[1]))
        left_col = min((c for r in range(start + 1, end) 
                        for c in range(input_grid.get_dimensions()[1]) 
                        if input_grid.values[r][c] == 1), default=None)
        right_col = max((c for r in range(start + 1, end) 
                         for c in range(input_grid.get_dimensions()[1]) 
                         if input_grid.values[r][c] == 1), default=None)
        
        if left_col is not None and right_col is not None:
            max_left_count = max(max_left_count, blue_count // 2 + blue_count % 2)
            section_info.append({
                'blue_count': blue_count,
                'left_col': left_col,
                'right_col': right_col
            })
        else:
            section_info.append(None)
    
    # Update left counts
    for info in section_info:
        if info:
            info['left_count'] = min(max_left_count, info['blue_count'])
            info['right_count'] = info['blue_count'] - info['left_count']
    
    return section_info

def rearrange_section(output_grid: ColoredGrid, start: int, end: int, info: dict):
    if not info:
        return
    
    available_space = end - start - 1
    rows_needed = max(info['left_count'], info['right_count'])
    empty_rows = available_space - rows_needed
    top_empty = empty_rows // 2
    bottom_empty = empty_rows - top_empty
    
    left_start = start + 1 + top_empty + rows_needed - info['left_count']
    right_start = start + 1 + top_empty + rows_needed - info['right_count']
    
    for i in range(info['left_count']):
        output_grid.values[left_start + i][info['left_col']] = 1
    
    for i in range(info['right_count']):
        output_grid.values[right_start + i][info['right_col']] = 1
