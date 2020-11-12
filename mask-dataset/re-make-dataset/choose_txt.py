import re
from numpy import *
import os

filePath='txt/'
file_list = os.listdir(filePath)

for file in file_list:
    fp = open("txt/"+file, 'r')
    op = open("result-nospace.txt",'a+')
    for eachline in fp.readlines():
        op.write(eachline.split(' ')[0]+'\n')
    # op.write('\n')
    
    fp.close()
    op.close()
