import pytest
from colored_grid import ColoredGrid


@pytest.fixture
def sample_grid():
    return ColoredGrid(values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_get_dimensions(sample_grid):
    assert sample_grid.get_dimensions() == (3, 3)


def test_get_cell(sample_grid):
    assert sample_grid.get_cell(1, 1) == 5


def test_set_cell(sample_grid):
    sample_grid.set_cell(1, 1, 0)
    assert sample_grid.get_cell(1, 1) == 0


def test_deep_copy(sample_grid):
    copied = sample_grid.deep_copy()
    assert copied.values == sample_grid.values
    copied.set_cell(0, 0, 0)
    assert copied.values != sample_grid.values


def test_rotate_90(sample_grid):
    rotated = sample_grid.rotate_90()
    assert rotated.values == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]


def test_flip_horizontal(sample_grid):
    flipped = sample_grid.flip_horizontal()
    assert flipped.values == [[3, 2, 1], [6, 5, 4], [9, 8, 7]]


def test_flip_vertical(sample_grid):
    flipped = sample_grid.flip_vertical()
    assert flipped.values == [[7, 8, 9], [4, 5, 6], [1, 2, 3]]


def test_count_color(sample_grid):
    assert sample_grid.count_color(5) == 1


def test_get_unique_colors(sample_grid):
    assert sample_grid.get_unique_colors() == set(range(1, 10))


def test_get_color_frequencies(sample_grid):
    assert sample_grid.get_color_frequencies() == {i: 1 for i in range(1, 10)}


def test_get_bounding_box(sample_grid):
    assert sample_grid.get_bounding_box(5) == (1, 1, 1, 1)


def test_find_connected_regions():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    regions = grid.find_connected_regions(2)
    assert len(regions) == 1, f"Expected 1 region, but found {len(regions)}"
    assert len(regions[0]) == 6, f"Expected region size 5, but found {len(regions[0])}"
    expected_region = [(0, 2), (1, 1), (2, 0), (1, 2), (2, 1), (2, 2)]
    assert set(regions[0]) == set(expected_region), f"Expected cells {expected_region}, but found {regions[0]}"


def test_find_connected_regions_complex():
    grid = ColoredGrid(
        values=[
            [1, 1, 2, 0, 2, 2],
            [1, 2, 2, 0, 1, 2],
            [2, 2, 1, 2, 2, 2],
            [1, 1, 2, 2, 1, 1],
            [2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 1],
        ]
    )
    for row in grid.values:
        print(row)

    regions = grid.find_connected_regions(2)

    assert len(regions) == 4, f"Expected 4 regions, but found {len(regions)}"
    regions.sort(key=len, reverse=True)

    assert len(regions[0]) == 8, f"Expected largest region size 8, but found {len(regions[0])}"
    expected_largest_region = {(0, 4), (0, 5), (1, 5), (2, 3), (2, 4), (2, 5), (3, 2), (3, 3)}
    assert (
        set(regions[0]) == expected_largest_region
    ), f"Largest region mismatch. Expected {expected_largest_region}, but found {set(regions[0])}"

    assert len(regions[1]) == 5, f"Expected second largest region size 5, but found {len(regions[1])}"
    expected_second_region = {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1)}
    assert (
        set(regions[1]) == expected_second_region
    ), f"Second largest region mismatch. Expected {expected_second_region}, but found {set(regions[1])}"

    assert len(regions[2]) == 4, f"Expected third region size 4, but found {len(regions[2])}"
    assert len(regions[3]) == 3, f"Expected fourth region size 3, but found {len(regions[3])}"


def test_get_symmetry_axes():
    # Both horizontal and vertical symmetry
    both_symmetric_grid = ColoredGrid(values=[[1, 2, 1], [3, 4, 3], [1, 2, 1]])
    assert both_symmetric_grid.get_symmetry_axes() == (True, True)

    # Only horizontal symmetry
    horizontal_symmetric_grid = ColoredGrid(values=[[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    assert horizontal_symmetric_grid.get_symmetry_axes() == (True, False)

    # Only vertical symmetry
    vertical_symmetric_grid = ColoredGrid(values=[[1, 2, 1], [3, 4, 3], [5, 6, 5]])
    assert vertical_symmetric_grid.get_symmetry_axes() == (False, True)

    # No symmetry
    non_symmetric_grid = ColoredGrid(values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert non_symmetric_grid.get_symmetry_axes() == (False, False)


def test_crop(sample_grid):
    cropped = sample_grid.crop(0, 0, 1, 1)
    assert cropped.values == [[1, 2], [4, 5]]


def test_expand(sample_grid):
    expanded = sample_grid.expand(1, 1, 1, 1, fill_color=0)
    assert expanded.values == [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]]


def test_replace_color(sample_grid):
    replaced = sample_grid.replace_color(5, 0)
    assert replaced.get_cell(1, 1) == 0


def test_apply_mask():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    mask = ColoredGrid(values=[[1, 0], [0, 1]])
    result = grid.apply_mask(mask, replace_color=0)
    assert result.values == [[1, 0], [0, 4]]


def test_find_pattern():
    grid = ColoredGrid(values=[[1, 2, 3], [4, 1, 2], [3, 4, 1]])
    pattern = ColoredGrid(values=[[1, 2], [4, 1]])
    assert grid.find_pattern(pattern) == [(0, 0), (1, 1)]


def test_extract_subgrid(sample_grid):
    subgrid = sample_grid.extract_subgrid(0, 0, 2, 2)
    assert subgrid.values == [[1, 2], [4, 5]]


def test_tile_grid():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    tiled = grid.tile_grid(2)
    assert tiled.values == [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]


def test_add():
    grid1 = ColoredGrid(values=[[1, 2], [3, 4]])
    grid2 = ColoredGrid(values=[[5, 6], [7, 8]])
    result = grid1.add(grid2)
    assert result.values == [[6, 8], [0, 2]]  # With default modulo 10


def test_subtract():
    grid1 = ColoredGrid(values=[[5, 6], [7, 8]])
    grid2 = ColoredGrid(values=[[1, 2], [3, 4]])
    result = grid1.subtract(grid2)
    assert result.values == [[4, 4], [4, 4]]


def test_multiply(sample_grid):
    result = sample_grid.multiply(2)
    assert result.values == [[2, 4, 6], [8, 0, 2], [4, 6, 8]]  # With default modulo 10


def test_from_subgrids():
    subgrid1 = ColoredGrid(values=[[1, 2], [3, 4]])
    subgrid2 = ColoredGrid(values=[[5, 6], [7, 8]])
    result = ColoredGrid.from_subgrids([[subgrid1, subgrid2]])
    assert result.values == [[1, 2, 5, 6], [3, 4, 7, 8]]


def test_split_into_subgrids(sample_grid):
    subgrids = sample_grid.split_into_subgrids(3, 1)
    assert len(subgrids) == 3
    assert all(len(row) == 1 for row in subgrids)
    assert subgrids[0][0].values == [[1, 2, 3]]


def test_to_binary(sample_grid):
    binary = sample_grid.to_binary(5)
    assert binary.values == [[0, 0, 0], [0, 1, 1], [1, 1, 1]]


def test_compress_rle(sample_grid):
    rle = sample_grid.compress_rle()
    assert rle == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]


def test_from_rle():
    rle = [(1, 2), (2, 2), (3, 2)]
    grid = ColoredGrid.from_rle(rle, width=2)
    assert grid.values == [[1, 1], [2, 2], [3, 3]]


def test_get_edge_cells(sample_grid):
    edges = sample_grid.get_edge_cells()
    assert set(edges) == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}


def test_flood_fill():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    filled = grid.flood_fill(0, 0, 3)
    assert filled.values == [[3, 3, 2], [3, 2, 2], [2, 2, 2]]


def test_detect_rectangles():
    grid = ColoredGrid(values=[[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    rectangles = grid.detect_rectangles()
    assert (1, 0, 0, 1, 1) in rectangles


def test_detect_lines():
    grid = ColoredGrid(values=[[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    lines = grid.detect_lines()
    assert (1, [(0, 1), (1, 1), (2, 1)]) in lines


def test_find_largest_object():
    grid = ColoredGrid(values=[[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    largest = grid.find_largest_object(1)
    assert set(largest) == {(0, 0), (0, 1), (1, 0)}


def test_diff():
    grid1 = ColoredGrid(values=[[1, 2], [3, 4]])
    grid2 = ColoredGrid(values=[[1, 3], [3, 4]])
    diff = grid1.diff(grid2)
    assert diff.values == [[0, 1], [0, 0]]


def test_similarity_score():
    grid1 = ColoredGrid(values=[[1, 2], [3, 4]])
    grid2 = ColoredGrid(values=[[1, 3], [3, 4]])
    score = grid1.similarity_score(grid2)
    assert score == 0.75


def test_scale():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    scaled = grid.scale(2)
    assert scaled.values == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


def test_find_repeating_pattern():
    grid = ColoredGrid(values=[[1, 2, 1, 2], [3, 4, 3, 4]])
    pattern = grid.find_repeating_pattern()
    assert pattern.values == [[1, 2], [3, 4]]


def test_extrapolate_sequence():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    extrapolated = grid.extrapolate_sequence("right", steps=1)
    assert extrapolated.values == [[1, 2, 2], [3, 4, 4]]


def test_find_center_of_mass(sample_grid):
    center = sample_grid.find_center_of_mass()
    assert center == pytest.approx((1.4, 1.1333333333333333))


def test_to_base64(sample_grid):
    base64_str = sample_grid.to_base64()
    decoded = ColoredGrid.from_base64(base64_str)
    assert decoded.values == sample_grid.values


def test_find_color_transitions(sample_grid):
    transitions = sample_grid.find_color_transitions()
    assert (0, 0, 1, 2) in transitions
    assert (0, 0, 1, 4) in transitions


def test_apply_function_to_regions():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    result = grid.apply_function_to_regions(lambda region: len(region))
    assert result.values == [[3, 3, 6], [3, 6, 6], [6, 6, 6]]


def test_apply_cellular_automaton():
    grid = ColoredGrid(values=[[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    def rule(neighbors):
        return 1 if sum(neighbors) >= 2 else 0

    result = grid.apply_cellular_automaton(rule)
    assert result.values == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def test_find_connected_regions_complex_exact():
    grid = ColoredGrid(
        values=[
            [1, 1, 2, 0, 2, 2],
            [1, 2, 2, 0, 1, 2],
            [2, 2, 1, 2, 2, 2],
            [1, 1, 2, 2, 1, 1],
            [2, 2, 1, 1, 2, 2],
            [2, 1, 1, 2, 2, 1],
        ]
    )
    regions = grid.find_connected_regions(2)
    regions.sort(key=len, reverse=True)
    assert set(regions[2]) == {(4, 4), (4, 5), (5, 3), (5, 4)}
    assert set(regions[3]) == {(4, 0), (4, 1), (5, 0)}


def test_extrapolate_sequence_all_directions():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    directions = ["right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"]
    for direction in directions:
        extrapolated = grid.extrapolate_sequence(direction, steps=1)
        assert extrapolated.get_dimensions() != (2, 2), f"Failed for direction: {direction}"


def test_empty_grid():
    with pytest.raises(ValueError):
        ColoredGrid(values=[])


def test_1x1_grid():
    grid = ColoredGrid(values=[[5]])
    assert grid.get_dimensions() == (1, 1)
    assert grid.find_connected_regions(5) == [[(0, 0)]]


def test_max_size_grid():
    max_grid = ColoredGrid(values=[[0] * 30 for _ in range(30)])
    assert max_grid.get_dimensions() == (30, 30)


def test_find_pattern_not_found():
    grid = ColoredGrid(values=[[1, 2, 3], [4, 5, 6]])
    pattern = ColoredGrid(values=[[7, 8], [9, 0]])
    assert grid.find_pattern(pattern) == []


def test_flood_fill_different_starts():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    filled1 = grid.flood_fill(0, 0, 3)
    filled2 = grid.flood_fill(2, 2, 3)
    assert filled1.values != filled2.values


def test_flood_fill_same_color():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    filled = grid.flood_fill(2, 2, 2)
    assert filled.values == grid.values


def test_detect_multiple_rectangles():
    grid = ColoredGrid(values=[[1, 1, 0, 2, 2], [1, 1, 0, 2, 2], [0, 0, 0, 0, 0], [3, 3, 3, 0, 0], [3, 3, 3, 0, 0]])
    rectangles = grid.detect_rectangles()
    expected_rectangles = [
        (1, 0, 0, 1, 1),  # (color, top, left, bottom, right)
        (2, 0, 3, 1, 4),
        (3, 3, 0, 4, 2),
    ]
    assert set(rectangles) == set(expected_rectangles)


def test_detect_multiple_lines():
    grid = ColoredGrid(values=[[1, 0, 2, 0, 3], [1, 0, 2, 0, 3], [1, 0, 2, 0, 3], [0, 0, 0, 0, 0], [4, 4, 4, 4, 4]])
    lines = grid.detect_lines()
    assert len(lines) == 4


def test_similarity_score_identical():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    assert grid.similarity_score(grid) == 1.0


def test_similarity_score_different():
    grid1 = ColoredGrid(values=[[1, 2], [3, 4]])
    grid2 = ColoredGrid(values=[[5, 6], [7, 8]])
    assert grid1.similarity_score(grid2) == 0.0


def test_find_center_of_mass_uniform():
    grid = ColoredGrid(values=[[1, 1], [1, 1]])
    center = grid.find_center_of_mass()
    assert center == (0.5, 0.5)


def test_find_center_of_mass_weighted():
    grid = ColoredGrid(values=[[1, 1], [1, 9]])
    center = grid.find_center_of_mass()
    assert center[0] > 0.5 and center[1] > 0.5


def test_apply_function_to_regions_color_change():
    grid = ColoredGrid(values=[[1, 1, 2], [1, 2, 2], [2, 2, 2]])
    result = grid.apply_function_to_regions(lambda region: len(region) % 5)
    assert result.values != grid.values


def test_get_cell_out_of_bounds():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        grid.get_cell(2, 0)


def test_set_cell_invalid_color():
    grid = ColoredGrid(values=[[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        grid.set_cell(0, 0, 10)


def test_compress_rle_complex():
    grid = ColoredGrid(values=[[1, 1, 1, 2, 2, 3, 3, 3, 3]])
    rle = grid.compress_rle()
    assert rle == [(1, 3), (2, 2), (3, 4)]


def test_from_rle_complex():
    rle = [(1, 3), (2, 2), (3, 4)]
    grid = ColoredGrid.from_rle(rle, width=9)
    assert grid.values == [[1, 1, 1, 2, 2, 3, 3, 3, 3]]


def test_find_repeating_pattern_none():
    grid = ColoredGrid(values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pattern = grid.find_repeating_pattern()
    assert pattern is None or pattern.values == grid.values


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
