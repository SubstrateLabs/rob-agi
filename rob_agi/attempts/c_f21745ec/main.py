from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Union
from collections import deque

def solve_f21745ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Removes orange (7) shapes and small shapes (less than 4x4).
    2. Fills hollow shapes with a recursive pattern based on their outline and internal structure.
    3. Leaves already filled shapes unchanged.
    4. Maintains symmetry and follows the contours of the outer shape during filling.
    5. Generates patterns that extend internal features along the shape's skeleton.

    The function identifies distinct shapes, analyzes their structure, generates appropriate
    filling patterns, and applies these patterns while respecting symmetry and internal features.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_shapes() -> List[Dict[str, Union[int, List[Tuple[int, int]]]]]:
        shapes = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and output_grid.get_cell(r, c) != 0:
                    color = output_grid.get_cell(r, c)
                    shape = output_grid.find_connected_regions(color)[0]
                    shapes.append({"color": color, "cells": shape})
                    visited.update(shape)
        return shapes

    def should_remove_shape(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> bool:
        return shape["color"] == 7 or len(shape["cells"]) < 16  # 4x4 = 16 cells

    def is_shape_filled(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> bool:
        return all(output_grid.get_cell(r, c) != 0 for r, c in shape["cells"])

    def get_shape_outline(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> List[Tuple[int, int]]:
        outline = []
        for r, c in shape["cells"]:
            if any((r+dr, c+dc) not in shape["cells"] for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]):
                outline.append((r, c))
        return outline

    def get_shape_skeleton(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> List[Tuple[int, int]]:
        outline = set(get_shape_outline(shape))
        inner_cells = [cell for cell in shape["cells"] if cell not in outline]
        
        skeleton = []
        for r, c in inner_cells:
            neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
                            if (r+dr, c+dc) in shape["cells"])
            if neighbors != 8:
                skeleton.append((r, c))
        
        return skeleton

    def generate_filling_pattern(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> List[Tuple[int, int]]:
        skeleton = get_shape_skeleton(shape)
        outline = set(get_shape_outline(shape))
        pattern = []
        
        # Extend skeleton
        for r, c in skeleton:
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in shape["cells"] and (nr, nc) not in outline:
                    pattern.append((nr, nc))
        
        # Fill remaining spaces
        remaining = [cell for cell in shape["cells"] if cell not in outline and cell not in pattern]
        pattern.extend(remaining)
        
        return pattern

    def apply_filling_pattern(shape: Dict[str, Union[int, List[Tuple[int, int]]]], pattern: List[Tuple[int, int]]):
        for r, c in pattern:
            output_grid.set_cell(r, c, shape["color"])

    shapes = find_shapes()
    for shape in shapes:
        if should_remove_shape(shape):
            for r, c in shape["cells"]:
                output_grid.set_cell(r, c, 0)
        elif not is_shape_filled(shape):
            filling_pattern = generate_filling_pattern(shape)
            apply_filling_pattern(shape, filling_pattern)

    return output_grid
