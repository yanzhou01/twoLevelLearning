## get items used in each coordinate
#!/usr/bin/env python
# coding: utf-8

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


def get_soup(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    return soup

def get_item_list (url):
    soup1 = get_soup(url)
    tags = soup1.find("ul", attrs={"class": "useitem_taglist"})
    item_brand = []
    item_link = []
    for tag in tags.find_all("li"):
        item_brand.append(tag.a.span.get_text())
        item_link.append(tag.a.get("href"))
    return item_brand, item_link

def get_item_img(url):
    soup1 = get_soup(url)
    tags = soup1.find("section", attrs={"class": "content_bg", "id":"item"})
    item_img = []
    for tag in tags.find_all("ul"):
        for tag in tags.find_all("li"):
            #print(tag)
            tag = tag.find("p", attrs={"class":"img"})
            item_img.append(tag.img.get("src"))
    return item_img

        

item_list_dict = {}
file_name = "item_list_dict"
for i in range(len(url)):
    print(i)
    try:
        item_list_dict[url[i]] = []
        item_list_dict[url[i]].append(get_item_list(url[i]))
        with open(file_name, mode='wb') as f:
            pickle.dump(item_list_dict, f)
        time.sleep(1)
    except:
        continue


def download(url):
    img_list = []
    img_lists = []
    for i in range(len(url)):
        try:
            img_list = get_item_img(url[i])
            img_lists.append(img_list)
            for j in range(len(img_list)):
                href_str = "https:" + img_list[j]
                print(i, j)
                download_img(href_str, '/Volumes/GoogleDrive/我的云端硬盘/fashion_research/item_img/%d_%d.jpg' %(i, j))
            img_list=[]
        except:
            continue
    return img_lists

img_lists = download(url)

def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)