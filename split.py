import re
import os
import linecache

# 本程序用于将widerface数据集中label部分分离出来并且重新保存
FILEDIR = "~\wider_face_split\\"  
file = open(FILEDIR+'wider_face_train_bbx_gt.txt', 'r')

def count_lines(file):
    lines_quantity = 0
    while True:
        buffer = file.read(1024 * 8192)  # 读取字节数
        if not buffer:
            break
        lines_quantity += buffer.count('\n')
    file.close()
    return lines_quantity


lines = count_lines(file)

for i in range(lines):
    line = linecache.getline(FILEDIR+'wider_face_val_bbx_gt.txt',i)
    if re.search('jpg', line):
        position = line.index('/')
        file_name = line[position + 1: -5]
        folder_name = line[:position]
        print(file_name)
        i += 1
        face_count = int(linecache.getline(FILEDIR+'wider_face_val_bbx_gt.txt', i))
        for j in range(face_count):
            # x1, y1, w, h, x1,y1 为人脸框左上角的坐标
            box_line = linecache.getline(FILEDIR + 'wider_face_val_bbx_gt.txt', i+j+1)
            po_x1 = box_line.index(' ')
            x1 = box_line[:po_x1]
            po_y1 = box_line.index(' ', po_x1 + 1)
            y1 = box_line[po_x1:po_y1]
            po_w = box_line.index(' ', po_y1 + 1)
            w = box_line[po_y1:po_w]
            po_h = box_line.index(' ', po_w + 1)
            h = box_line[po_w:po_h]
            coordinates = x1 + y1 + w + h
            # print(coordinates)
            if not(os.path.exists(FILEDIR + "wider_face_val\\" + folder_name)):
                os.makedirs(FILEDIR + "wider_face_val\\" + folder_name)
            with open(FILEDIR + "wider_face_val\\"+ folder_name + "\\" + file_name + ".txt", 'a') as f:
                f.write(coordinates + "\n")
            f.close()
        i += i + j + 1
