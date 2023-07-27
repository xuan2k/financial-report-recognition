# from typing import List

# def calculate_intersection(box1, box2):
#     # Function to calculate the intersection area of two boxes
#     # Assuming boxes are in the format [x1, y1, x2, y2]
#     x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
#     y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
#     return x_overlap * y_overlap

# def find_best_match_boxes(cell, text_boxes):
#     # Function to find the best matching text boxes for a given cell
#     best_matches = []
#     for text_box in text_boxes:
#         intersection_area = calculate_intersection(cell, text_box)
#         overlap_percentage = intersection_area / ((cell[2] - cell[0]) * (cell[3] - cell[1]))
#         best_matches.append((text_box, overlap_percentage))
#     best_matches.sort(key=lambda x: x[1], reverse=True)
#     return best_matches

# def find_closest_text_box(empty_cell, text_boxes, non_empty_cells):
#     # Function to find the text box closest to the center of the empty cell
#     empty_cell_center = [(empty_cell[0] + empty_cell[2]) / 2, (empty_cell[1] + empty_cell[3]) / 2]
#     min_distance = float('inf')
#     closest_text_box = None

#     for neighbor_cell in non_empty_cells:
#         neighbor_cell_center = [(neighbor_cell[0] + neighbor_cell[2]) / 2, (neighbor_cell[1] + neighbor_cell[3]) / 2]
#         distance = ((empty_cell_center[0] - neighbor_cell_center[0]) ** 2) + ((empty_cell_center[1] - neighbor_cell_center[1]) ** 2)
#         if distance < min_distance:
#             min_distance = distance
#             closest_text_box = find_best_match_boxes(neighbor_cell, text_boxes)[0][0]  # Get the best matching text box for the neighbor cell

#     return closest_text_box

# def match_boxes(list_a: List[List[int]], list_b: List[List[int]], text_b: List[str]) -> List[List[List[int]]]:
#     result = []

#     non_empty_cells = [cell for cell in list_a if cell]

#     for cell in list_a:
#         if not cell:
#             # closest_text_box = find_closest_text_box(cell, list_b, non_empty_cells)
#             # if closest_text_box:
#             #     result.append([closest_text_box])
#             # else:
#             #     result.append([])
#             result.append([[]])
#             # pass
#         else:
#             best_matches = find_best_match_boxes(cell, list_b)
#             cell_result = []
#             used_text_boxes = set()
#             for text_box, overlap_percentage in best_matches:
#                 if overlap_percentage == 0:
#                     break

#                 if tuple(text_box) not in used_text_boxes:  # Convert text_box to tuple here
#                     cell_result.append(text_box)
#                     used_text_boxes.add(tuple(text_box))  # Convert text_box to tuple here

            
#             result.append(cell_result)

#     return result

# # Example usage
# list_a = [[], [24, 1, 56, 74], [723, 1, 824, 74], [867, 3, 1020, 74], [1091, 1, 1237, 34], 
# 	  [], [], [], [], [1091, 36, 1239, 70], 
# 	  [], [96, 82, 626, 125], [], [], []]
# list_b = [[727, 0, 827, 31], [890, 0, 1021, 31], [1095, 0, 1241, 31], 
# 	  [734, 37, 814, 70], [879, 37, 1021, 70], [1094, 29, 1242, 74], 
# 	  [18, 80, 68, 129], [81, 86, 205, 119]]
# text_b = ["Thuyết", "31/3/2018", "31/12/2017", "minh", "Triệu đồng", "Triệu đồng", []]

# output = match_boxes(list_a, list_b, text_b)
# for i, item_1 in enumerate(output):
#     for j, item_2 in enumerate(output):
#         if len(item_1) == 0 or len(item_2) == 0:
#             continue
#         for list in item_1:
#             if list in item_2:
#                 if len(item_1) > len(item_2):
#                     item_2.remove(list)
# print(output)

import time
def number_processing(txt):
    removal = ['(',')','.',',']
    for c in removal:
        txt = txt.replace(c, "")
    if txt.isdigit():
        pos = 1
        insert = 0
        while len(txt) > (3*pos + insert):
            #1234 -> 1.234: len = 4, 3*pos = 3 -> input '.' to str[-3]
            txt = txt[:len(txt)-3*pos-insert] + '.' + txt[-3*pos-insert:]
            pos += 1
            insert += 1
        return txt
    return None

print(number_processing('a'))