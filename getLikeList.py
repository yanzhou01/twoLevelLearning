## get likes info of each coordinate
# coding: UTF-8
import time
import sys
import requests
from bs4 import BeautifulSoup
import re
import os
import shutil
import numpy as np
import pandas as pd
import pickle


file_name = '/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes/features.pickle'
with open(file_name, 'rb') as handle:
    tempDictFeatures = pickle.load(handle)

data = pd.DataFrame.from_dict(tempDictFeatures, orient = 'index')
url = data.url

def get_soup_like(url, j = 0):
    req = requests.get(url+"like/?pageno=" + str(j))
    soup = BeautifulSoup(req.content, 'html.parser')
    return soup

     
def get_max_page(url):
    soup1 = get_soup_like(url)
    page = soup1.find(id = "pager_container")
    page.find_all("li")
    tmp = []
    for page in page.find_all("li"):
        tmp.append(page.get_text())
    #print(tmp)
    max_page = int(tmp[-1])
    return max_page


def get_like_list(url, j):
    soup1 = get_soup_like(url, j)
    likes = soup1.find_all("h3")
    like_list = []
    for like in likes:
        try:
            like_list.append(like.a.get("href"))
        except:
            continue
    return like_list


file_name = "/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes/like_dict_from_5958.pickle"
like_dict={}
for i in range(5958, len(url)):
    
    try:
        max_page = get_max_page(url[i]) #then go next iterative? 
        time.sleep(1)
        
        print("i=" + str(i))
        like_dict[url[i]]=[]

        for j in range(max_page):
            like_list = get_like_list(url[i], j)
            like_dict[url[i]].extend(like_list) # should append elements
            #print(like_list)
            print("\t j="+str(j))
            #time.sleep(1)
        with open(file_name, mode='wb') as f:
            pickle.dump(like_dict, f)
    except:
        continue