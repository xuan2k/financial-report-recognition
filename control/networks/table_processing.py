import cv2
import copy
import json
import math
import yaml
import operator
import numpy as np  
import pandas as pd
from typing import List
from functools import reduce
from openpyxl import Workbook
from bs4 import BeautifulSoup
from unidecode import unidecode
# from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from .box import *

class TableProcessing:
    def __init__(self, config) -> None:
        self.copus = {}
        self.tfidf_matrix = {}

        self.config = {}
        self.vectorizer = {}

        if isinstance(config, str):
            with open(config, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.config.update(config)

        elif isinstance(config, dict):    
            self.config.update(config)
        else:
            raise TypeError("Unsupport config input type!")
        
        self.c_score = {}
        for k,v in self.config.items():
            self.vectorizer[k] = TfidfVectorizer()
            doc = []
            for item in open(v, 'r'):
                title = item.split(' ')[0]
                item = item.replace(title, '')
                item = item.strip(' ')
                item = item.strip('\n')
                doc.append(item)
            
            self.copus[k] = doc
            self.c_score[k] = 0.0

            documents = self.copus[k]

            preprocessed_documents = [unidecode(doc.lower()) for doc in documents]


            curr_tfidf_matrix = self.vectorizer[k].fit_transform(preprocessed_documents)

            self.tfidf_matrix[k] = curr_tfidf_matrix

    def isnum(self, txt):
        removal = ['(',')','.',',']
        tmp = txt
        for c in removal:
            tmp = tmp.replace(c, "")
        if tmp.isdigit():
            return True
        else:
            return False

    def empty_row(self, lst):
        for item in lst:
            for i in item:
                if i != '':
                    return False
                
        return True

    def list_num(self, lst):
        for i in lst:
            if not (self.isnum(i)):
                return False
        
        return True

    def remove_empty(self, lst:list):
        res = []
        for item in lst:
            if len(item) == 1:
                res.append(item)
                continue
            tmp = [i for i in item if i != '']
            res.append(tmp)
        return res
    
    def number_processing(self, txt):
        removal = ['(',')','.',',']
        neg = False
        for c in removal:
            if '(' in txt or ')' in txt:
                neg = True
            txt = txt.replace(c, "")
        if txt.isdigit():
            if txt[0] == '0':
                txt = txt[1:]
            pos = 1
            insert = 0
            while len(txt) > (3*pos + insert):
                #1234 -> 1.234: len = 4, 3*pos = 3 -> input '.' to str[-3]
                txt = txt[:len(txt)-3*pos-insert] + '.' + txt[-3*pos-insert:]
                pos += 1
                insert += 1
            if neg:
                txt = '(' + txt + ')'
            return txt
        return None
    
    def calculate_intersection(self, box1, box2):
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        return x_overlap * y_overlap
    
    def find_best_match_boxes(self, cell, text_boxes, content):
        best_matches = []
        for i, text_box in enumerate(text_boxes):
            intersection_area = self.calculate_intersection(cell, text_box)
            overlap_percentage = intersection_area / ((text_box[2] - text_box[0]) * (text_box[3] - text_box[1]))
            best_matches.append((text_box, overlap_percentage, content[i]))
        return best_matches

    def find_closest_text_box(self, curr_box, all_box):
        # Function to find the text box closest to the center of the empty cell
        curr_box_center = [(curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2]
        min_distance = float('inf')
        closest_text_box = None
        closest_center = []
        for neighbor_box in all_box:
            neighbor_cell_center = [(neighbor_box[0] + neighbor_box[2]) / 2, (neighbor_box[1] + neighbor_box[3]) / 2]
            distance = ((curr_box_center[0] - neighbor_cell_center[0]) ** 2) + ((curr_box_center[1] - neighbor_cell_center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_text_box = neighbor_box

                line = [neighbor_cell_center[0]-curr_box_center[0], neighbor_cell_center[1]-curr_box_center[1]]
                closest_center = neighbor_cell_center

        cen_dist = math.sqrt(line[0]**2 + line[1]**2)/(math.sqrt((curr_box[0]-curr_box_center[0])**2 + (curr_box[1]-curr_box_center[1])**2)+ math.sqrt((closest_text_box[0]-closest_center[0])**2 + (closest_text_box[1]-closest_center[1])**2))

        cosine = line[0]/math.sqrt(line[0]**2 + line[1]**2)
        upper = 0
        left = 0
        if abs(cen_dist) < 0.9 or cosine < 0.7:
            if line[1] > 0:
                upper = 1
            else:
                upper = -1
        else:
            if line[0] > 0:
                left = 1
            else:
                left = -1

        return (closest_text_box, upper, left)

    def match_boxes(self, list_a: List[List[int]], list_b: List[List[int]], text_b: List[str], num_col) -> List[List[List[int]]]:
        result = []
        res_box = []
        used_text_boxes = {}
        idx = 0

        for cell in list_a:
            if not cell:
                result.append([''])
                res_box.append([[]])
                idx += 1
            else:
                best_matches = self.find_best_match_boxes(cell, list_b, text_b)
                cell_result = []
                cell_res_box = []

                for text_box, overlap_percentage, content in best_matches:
                    if overlap_percentage < 0.5:
                        continue
                    if tuple(text_box) not in used_text_boxes.keys():  # Convert text_box to tuple here
                        cell_result.append(content)
                        cell_res_box.append(text_box)
                        used_text_boxes[tuple(text_box)] = (overlap_percentage, idx, content)  # Convert text_box to tuple here
                    else:
                        if used_text_boxes[tuple(text_box)][0] < overlap_percentage:
                            id = used_text_boxes[tuple(text_box)][1]
                            nid = result[id].index(content)
                            result[id][nid] = ""
                            res_box[id][nid] = []
                            cell_result.append(text_box)
                            cell_res_box.append(text_box)
                            used_text_boxes[tuple(text_box)] = (overlap_percentage, idx, content)  # Convert text_box to tuple here

                if len(cell_result) == 0:
                    cell_result = ['']
                    cell_res_box = [[]]
                    

                result.append(cell_result)
                res_box.append(cell_res_box)
                idx += 1
        # num_ele = len(result)
        # columns = None
        # for i in range(num_ele-num_col):
        #     if i%5 ==0:
        #         print(result[i:i+num_col])
        #         print(res_box[i:i+num_col])
        # print(result)
        used = used_text_boxes.keys()
        used = [list(item) for item in used]
        for i, item in enumerate(list_b):
            if item not in used:
                out = self.find_closest_text_box(item, used)
                # print(out, used_text_boxes[tuple(out[0])],text_b[i], item)
                id = used_text_boxes[tuple(out[0])][1]
                content = used_text_boxes[tuple(out[0])][2]
                nid = result[id].index(content)
                upper = out[1]
                left = out[2]
                if upper != 0:
                    new_idx = id-upper*num_col
                    result[new_idx].append(text_b[i])

                if left != 0:
                    new_idx = id-left
                    result[new_idx].append(text_b[i])
                
                used.append(item)
                used_text_boxes[tuple(item)] = (1, new_idx, text_b[i])
            # else:
            #     print(f"Detected {text_b[i], used_text_boxes[tuple(item)]}")

        return result

    # def merge_row(self, row1, row2):
        
    def similarity(self, text_list):
        self.c_score = {}
        res_txt = {}
        for k,v in self.tfidf_matrix.items():
            new_doc = []
            score= 0.0
            for texts in text_list: 
                tmp_doc = []
                for item in texts:
                    if self.isnum(item):
                        tmp_doc.append(item)
                        continue
                    process_inp = unidecode(item.lower())
                    new_tfidf = self.vectorizer[k].transform([process_inp])
                    similarity_score = cosine_similarity(new_tfidf, v)
                    best_score = np.max(similarity_score[0])
                    if best_score < 0.5:
                        tmp_doc.append(item)
                    else:               
                        score += best_score
                        tmp_doc.append(self.copus[k][np.argmax(similarity_score)])
                new_doc.append(tmp_doc)
            res_txt[k] = new_doc
            self.c_score[k] = score
        # print(self.c_score, max(self.c_score.items(), key=operator.itemgetter(1))[0])
        best_match = max(self.c_score.items(), key=operator.itemgetter(1))[0]
        return res_txt[best_match]

    def old_box_mapping(self, table_data, cellss):
        merge= []
        for i, cells in enumerate(cellss):
            for j, cell in enumerate(cells):
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                merge_row = rowspan > 1
                merge_col = colspan > 1
                merge_mix = rowspan > 1 and colspan > 1

                if merge_mix:
                    for h in range(0, rowspan):
                        for k in range(1, colspan):
                            if h != 0 and k == 1:
                                table_data[i + h].insert(j, [])   

                            table_data[i + h].insert(j+k, [])   

                elif merge_row:
                    for row in table_data[i+1:i+rowspan]:
                        row.insert(j, [])
                    merge.append([i+1,j+1,i+rowspan,j+1])
                        
                elif merge_col:
                    for k in range(1, colspan):
                        table_data[i].insert(j+k, [])   
                    merge.append([i+1,j+1,i+1,j+colspan])
            # print(table_data[i])
        return table_data

    def box_mapping(self, table_data):
        merge= []
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                merge_row = isinstance(cell, tuple) and cell[2] > 1
                merge_col = isinstance(cell, tuple) and cell[1] > 1
                merge_mix = isinstance(cell, tuple) and cell[1] > 1 and cell[2] > 1

                if merge_mix:
                    # tmp_item = (table_data[i][j], colspan, rowspan) 
                    table_data[i][j] = cell[0]
                    for h in range(0, cell[2]):
                        for k in range(1, cell[1]):
                            if h != 0 and k == 1:
                                table_data[i + h].insert(j, [])   

                            table_data[i + h].insert(j+k, [])   
                    # self.box_alignment(table_data, (i, j))
                    
                elif merge_row:
                    # tmp_item = (table_data[i][j], 0, rowspan) 
                    table_data[i][j] = cell[0]
                    for row in table_data[i+1:i+cell[2]]:
                        row.insert(j, [])
                    merge.append([i+1,j+1,i+cell[2],j+1])
                    # self.box_alignment(table_data, (i, j))
                        
                elif merge_col:
                    # tmp_item = (table_data[i][j], colspan, 0) 
                    table_data[i][j] = cell[0]
                    for k in range(1, cell[1]):
                        table_data[i].insert(j+k, [])   
                    merge.append([i+1,j+1,i+1,j+cell[1]])
                # else:
                #     tmp_item = table_data[i][j]
                # tmp_row.append(tmp_item)
            # tmp_tab.append(tmp_row)
            # print(table_data[i])
            # print(tmp_tab[i])
        return table_data

    def row_mapping(self, rows, bboxes):
        # Loop through each row in the table
        table_data = []
        cellss = []
        count = 0
        for row in rows:
            # Find all the cells in the row
            cells = row.find_all(['td', 'th'])
            row_data = []
            # print(cells)
            for i, cell in enumerate(cells):
                try:
                    row_data.append(bboxes[count+i])
                except:
                    row_data.append([])
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                merge_row = rowspan > 1
                merge_col = colspan > 1
                merge_mix = rowspan > 1 and colspan > 1

                if merge_mix:
                    row_data[i] = (row_data[i], colspan, rowspan) 

                elif merge_row:
                    row_data[i] = (row_data[i], 0, rowspan) 
                        
                elif merge_col:
                    row_data[i] = (row_data[i], colspan, 0) 

            count += len(cells)
            # print(row_data, count, len(bboxes))
            table_data.append(row_data)
        return table_data

    def old_row_mapping(self, rows, bboxes):
        # Loop through each row in the table
        table_data = []
        cellss = []
        count = 0
        for row in rows:
            # Find all the cells in the row
            cells = row.find_all(['td', 'th'])
            cellss.append(cells)
            
            # Extract the content of each cell
            row_data = [(bboxes[count + i]) for i in range(len(cells))]
            count += len(cells)
            table_data.append(row_data)

    def box_alignment(self, table_data, location):
        i, j = location
        cell = table_data[i][j]
        top_constrain = []
        bot_constrain = []
        for row in range(i, i + cell[2]):
            # for cell in row:
            for item in table_data[row]:
                if row == i and len(item) > 0:
                    top_constrain.append(item)
                # print(f"processing for row {item}")
        table_data[i][j] = cell[0]
        return table_data



    def to_excel(self, data, rec):

        wb = Workbook()
        ws = wb.active

        html_table = data["html"]
        bboxes = data["content_ann"]["bboxes"]
    
        rec_boxes = rec["bboxes"]

        for i, txt in enumerate(rec["text"]):
            t_res = self.number_processing(txt)
            if t_res:
                rec["text"][i] = t_res 
        # Parse the HTML table using BeautifulSoup
        soup = BeautifulSoup(html_table, 'html.parser')

        # Find all the rows in the table
        rows = soup.find_all('tr')

        # Initialize an empty list to store the table data
        list_a = []

        table_data = self.row_mapping(rows, bboxes)
        table_data = self.box_mapping(table_data)
        # print(table_data)
        # exit()

        # table_data = self.box_mapping(table_data, cellss)

        num_col = len(table_data[0])

        list_a = [item for row in table_data for item in row ]
        out = self.match_boxes(list_a, rec_boxes, rec["text"], num_col)
        n_row = len(table_data[0])
        content_data = []
        remove_flag = True
        for i, row in enumerate(table_data):
            row = out[n_row*i:n_row*i+n_row]
            # print(row)
            row = self.remove_empty(row)
            for items in row:
                for item in items:
                    if "Thuyết" in item:
                        remove_flag = False
            if remove_flag:
                continue
            if not self.empty_row(row):
                content_data.append(row)
        content_tmp_data = []
        for i, row in enumerate(content_data):
            # print(row)
            row = self.remove_empty(row)
            tmp = row
            row = []
            for j, item in enumerate(tmp):
                if len(item) > 1:
                    if self.list_num(item):
                        n = len(item)
                        for k in range(1, n):
                            # print(f"check {i,j,k, len(content_data)}")
                            content_data[i+k][j].append(item[k]) 
                        item = item[0]
                        # pass
                    else:
                        item = reduce(lambda x, y: x + " " + y if len(x) > 0 and len(y) > 0 else x + y, item)
                else:
                    item = item[0]
                row.append(item)
            rel = row[0] + " " + row[1]
            row[0] = (rel).replace(rel.split(' ')[0], "").strip(" ") if len(row[0]) > 0 else row[1].strip(" ")
            row.remove(row[1])
            content_tmp_data.append(row)
        for i, row in enumerate(content_tmp_data):
            if self.isnum(row[-1] + row[-2]) and row[0]=='':
                new_row = []
                for j, text in enumerate(content_tmp_data[i-1]):
                    new_row.append(text + row[j])
                    content_tmp_data[i-1][j] = ''
                content_tmp_data[i] = new_row
            # header = row[0]
            # print(self.list_num(row[-1]+row[-2]), row[-1]+row[-2])
            # if header == '' and self.isnum(row[-1]+row[-2]):
            #     print("check")
            #     for j in range(1, content_data[i-1]):
            #         print(f"Check {i}, {j}, {len(content_data[i-1])}, {len(row)}")
            #         content_data[i-1][j] += (row[j]) 
        # print(content_tmp_data)
        processed_txt = self.similarity(content_tmp_data)
        # print(processed_txt)
        for row in processed_txt:
        # for row in content_tmp_data:
            # print(row)
            if ''.join(row)!='':
                ws.append(row)


        # Extend the cell to fit the content
        for column_cells in ws.columns:
            max_length = 0
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            ws.column_dimensions[column_cells[0].column_letter].width = adjusted_width

        return wb

if __name__ == "__main__":
    ver = 3
    for i in range(6):
        ver = i+1
        print(f"run {ver}")
        excel_file = f'/home/xuan/Project/OCR/demo/pred_1/demo_{ver}.xlsx'

        f = open(f"/home/xuan/Project/OCR/demo/pred_1/res_{ver}_TSR.txt", 'r')
        rec_f = open(f"/home/xuan/Project/OCR/demo/text_1/{ver}/res.json", "r")

        config = f"../configs/model/table_processing/config.yml"

        t = TableProcessing(config)
        data = json.load(f)
        rec = json.load(rec_f)
        res = t.to_excel(data, rec)
        res.save(excel_file)

        print(f"DataFrame saved to '{excel_file}' with cell merging.")
