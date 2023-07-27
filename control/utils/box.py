from typing import List

def calculate_intersection(box1, box2):
    # Function to calculate the intersection area of two boxes
    # Assuming boxes are in the format [x1, y1, x2, y2]
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    return x_overlap * y_overlap

def find_best_match_boxes(cell, text_boxes, content):
    # Function to find the best matching text boxes for a given cell
    best_matches = []
    # print(len(text_boxes), len(content))
    for i, text_box in enumerate(text_boxes):
        intersection_area = calculate_intersection(cell, text_box)
        overlap_percentage = intersection_area / ((cell[2] - cell[0]) * (cell[3] - cell[1]))
        best_matches.append((text_box, overlap_percentage, content[i]))
    # best_matches.sort(key=lambda x: x[1], reverse=True)
    # print(best_matches)
    return best_matches

def find_closest_text_box(empty_cell, text_boxes, non_empty_cells):
    # Function to find the text box closest to the center of the empty cell
    empty_cell_center = [(empty_cell[0] + empty_cell[2]) / 2, (empty_cell[1] + empty_cell[3]) / 2]
    min_distance = float('inf')
    closest_text_box = None

    for neighbor_cell in non_empty_cells:
        neighbor_cell_center = [(neighbor_cell[0] + neighbor_cell[2]) / 2, (neighbor_cell[1] + neighbor_cell[3]) / 2]
        distance = ((empty_cell_center[0] - neighbor_cell_center[0]) ** 2) + ((empty_cell_center[1] - neighbor_cell_center[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_text_box = find_best_match_boxes(neighbor_cell, text_boxes)[0][0]  # Get the best matching text box for the neighbor cell

    return closest_text_box

def match_boxes(list_a: List[List[int]], list_b: List[List[int]], text_b: List[str]) -> List[List[List[int]]]:
    result = []

    non_empty_cells = [cell for cell in list_a if cell]
    used_text_boxes = {}
    idx = 0
    # used_text_boxes = set()

    # used_text_boxes = set()
    for cell in list_a:
        if not cell:
            # closest_text_box = find_closest_text_box(cell, list_b, non_empty_cells)
            # if closest_text_box:
            #     result.append([closest_text_box])
            # else:
            #     result.append([])
            result.append([''])
            idx += 1
            # result.append([[]])
            # pass
        else:
            best_matches = find_best_match_boxes(cell, list_b, text_b)
            cell_result = []
            # used_text_boxes = set()

            for text_box, overlap_percentage, content in best_matches:
                if overlap_percentage == 0:
                    continue
                # print(f"Check {text_box} and {used_text_boxes}")
                if tuple(text_box) not in used_text_boxes.keys():  # Convert text_box to tuple here
                    # print(overlap_percentage)
                    cell_result.append(content)
                    used_text_boxes[tuple(text_box)] = (overlap_percentage, idx)  # Convert text_box to tuple here
                else:
                    if used_text_boxes[tuple(text_box)][0] < overlap_percentage:
                        id = used_text_boxes[tuple(text_box)][1]
                        # print(result, used_text_boxes[tuple(text_box)])
                        # print(result[id], content, sep=" ")
                        nid = result[id].index(content)
                        result[id][nid] = ""
                        cell_result.append(content)
                        used_text_boxes[tuple(text_box)] = (overlap_percentage, idx)  # Convert text_box to tuple here

                # else:
                #     result.append([''])

            if len(cell_result) == 0:
                cell_result = ['']
                

            result.append(cell_result)
            idx += 1
    # print(used_text_boxes)
    return result

def filter_output(output):
    for i, item_1 in enumerate(output):
        for j, item_2 in enumerate(output):
            if len(item_1) == 0 or len(item_2) == 0:
                continue
            for list in item_1:
                if list in item_2 and i != j:
                    if len(item_1) > len(item_2):
                        item_2.remove(list)

if __name__ == "__main__":
    # Example usage
    list_a = [[], [24, 1, 56, 74], [723, 1, 824, 74], [867, 3, 1020, 74], [1091, 1, 1237, 34], 
        [], [], [], [], [1091, 36, 1239, 70], 
        [], [96, 82, 626, 125], [], [], []]
    list_b = [[727, 0, 827, 31], [890, 0, 1021, 31], [1095, 0, 1241, 31], 
        [734, 37, 814, 70], [879, 37, 1021, 70], [1094, 29, 1242, 74], 
        [18, 80, 68, 129], [81, 86, 205, 119]]
    text_b = ["Thuyết", "31/3/2018", "31/12/2017", "minh", "Triệu đồng", "Triệu đồng", "A", "Tai san"]

    output = match_boxes(list_a, list_b, text_b)
    # filter_output(output)
    print(output)
