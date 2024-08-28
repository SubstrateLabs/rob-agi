from rob_agi.colored_grid import ColoredGrid

def print_column(column):
    color_map = {0: '⬛', 4: '🟨', 6: '🟪', 7: '🟧', 8: '🟦'}
    return ''.join(color_map.get(c, '⬜') for c in column)

def transform_column(column):
    yellow_positions = [i for i, c in enumerate(column) if c == 4]
    if len(yellow_positions) < 2:
        return column
    
    start, end = min(yellow_positions), max(yellow_positions)
    new_column = column.copy()
    
    for i in range(start + 1, end):
        if column[i] in [0, 6]:
            if i == start + 1 or i == end - 1:
                new_column[i] = 8  # sky next to yellow
            else:
                new_column[i] = 7  # orange in between
    
    return new_column

# Test cases
test_columns = [
    [0, 4, 0, 0, 6, 0, 4, 0],
    [4, 6, 6, 6, 6, 4],
    [0, 4, 6, 0, 6, 4, 0],
    [4, 0, 0, 0, 4],
    [0, 4, 6, 7, 6, 4, 0],
    [4, 6, 6, 6, 6, 6, 4],
]

print("Original | Transformed")
print("---------|------------")
for column in test_columns:
    transformed = transform_column(column)
    print(f"{print_column(column)} | {print_column(transformed)}")
