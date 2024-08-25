from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f21745ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Removes orange (7) shapes and small shapes (less than 5x5).
    2. Fills hollow shapes with a recursive pattern based on their outline.
    3. Leaves already filled shapes unchanged.
    4. Maintains symmetry and follows the contours of the outer shape during filling.

    The function identifies distinct shapes, processes each shape according to the rules,
    and returns the transformed grid.
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
        return shape["color"] == 7 or len(shape["cells"]) < 25  # 5x5 = 25 cells

    def is_shape_filled(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> bool:
        return all(output_grid.get_cell(r, c) != 0 for r, c in shape["cells"])

    def get_shape_outline(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> List[Tuple[int, int]]:
        outline = []
        for r, c in shape["cells"]:
            if any((r+dr, c+dc) not in shape["cells"] for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]):
                outline.append((r, c))
        return outline

    def find_fillable_regions(shape: Dict[str, Union[int, List[Tuple[int, int]]]]) -> List[List[Tuple[int, int]]]:
        outline = set(get_shape_outline(shape))
        inner_cells = [cell for cell in shape["cells"] if cell not in outline]
        
        regions = []
        visited = set()
        for cell in inner_cells:
            if cell not in visited and output_grid.get_cell(*cell) == 0:
                region = []
                stack = [cell]
                while stack:
                    r, c = stack.pop()
                    if (r, c) not in visited and (r, c) in inner_cells and output_grid.get_cell(r, c) == 0:
                        visited.add((r, c))
                        region.append((r, c))
                        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in inner_cells:
                                stack.append((nr, nc))
                if len(region) > 1:
                    regions.append(region)
        return regions

    def maintain_symmetry(shape: Dict[str, Union[int, List[Tuple[int, int]]]], regions: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        if not regions:
            return []
        
        # Find the center of the shape
        min_r = min(r for r, _ in shape["cells"])
        max_r = max(r for r, _ in shape["cells"])
        min_c = min(c for _, c in shape["cells"])
        max_c = max(c for _, c in shape["cells"])
        center_r, center_c = (min_r + max_r) // 2, (min_c + max_c) // 2
        
        # Choose the region closest to the center
        return min(regions, key=lambda region: min((r-center_r)**2 + (c-center_c)**2 for r, c in region))

    def recursive_fill(shape: Dict[str, Union[int, List[Tuple[int, int]]]]):
        while True:
            regions = find_fillable_regions(shape)
            if not regions:
                break
            best_region = maintain_symmetry(shape, regions)
            for r, c in best_region:
                output_grid.set_cell(r, c, shape["color"])

    shapes = find_shapes()
    for shape in shapes:
        if should_remove_shape(shape):
            for r, c in shape["cells"]:
                output_grid.set_cell(r, c, 0)
        elif not is_shape_filled(shape):
            recursive_fill(shape)

    return output_grid
