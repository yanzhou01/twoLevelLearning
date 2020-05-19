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

# get feature of one obs
import time
import sys
import requests
from bs4 import BeautifulSoup
import re  
import os
import shutil
import pandas as pd
import numpy as np


def get_soup(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    return soup


def text_with_newlines(elem):
    text = ''
    if not elem is None:
        for e in elem.recursiveChildGenerator():
            if isinstance(e, str):
                text += e.strip()
            elif e.name == 'br':
                text += ' '
    return text


def get_label(soup, features):
    temp = ""
    for a in soup.find_all("a", href=re.compile("/category/")):
        try:
            h1_text = a.string
            if (h1_text != "カテゴリ一覧"):
                if (not '×' in h1_text):
                    if (not 'その他' in h1_text):
                        #print(h1_text)
                        temp += h1_text + "\n"
                        features["coordinate"] = temp
        except Exception as e:
            print(e)
    return features


# get title, url, descriptions
def get_title_url(soup, features):
    for tag in soup.find_all("meta"):
        if tag.get("name", None) == "description":
            features["title"] = tag.get("content", None)
            #print(tag.get("content", None))
        elif tag.get("property", None) == "og:url":
            features["url"] = tag.get("content", None)
            #print(tag.get("content", None))
    features["description"] = text_with_newlines(soup.find(attrs={"class": "content_txt"}))
    return features


def get_favLikes(temp_soup, url, feature):
    favNum = temp_soup.find("p", {"class": "allBtn"})
    ##
    #print(favNum.getText())
    ##
    feature["fav"] = favNum.getText()

    likeNums = temp_soup.find_all("div", {"class": "btn_like fn_btn"})
    for likeNum in likeNums:
        urlID = re.findall("\/[0-9]+\/", url)
        if urlID != None:
            while likeNum['data-likeid'] in urlID[0]:
                ##
                #print(likeNum.getText())
                ##
                feature["like"] = likeNum.getText().strip()
                break
    return feature


def get_feature(url):
    soup_temp = get_soup(url)
    feature = {}
    feature = get_title_url(soup_temp, feature)
    feature = get_label(soup_temp, feature)
    feature = get_favLikes(soup_temp, url, feature)
    return feature


def download():
    global b, i
    for a_tag in soup.find_all("img"):
        try:
            href_str = a_tag.get("data-original")
            if b % 2 != 0:
                href_str = "http:" + href_str
                download_img(href_str, '%d.jpg' % i)
                #print(href_str)
                i += 1
            b += 1
            # resources.append(href_str)
        except Exception as e:
            print(e)
        finally:
            time.sleep(1)
    b = 0


def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)




i = 0
j = 1
features = {}
file_name = "features.pickle"
for k in [0, 1]:
    if k == 0:
        url_head = "https://wear.jp/women-coordinate/?pageno=%d"
    elif k == 1:
        url_head = "https://wear.jp/men-coordinate/?pageno=%d"
    k += 1
    for j in range(1, 222):
        print("j: " + str(j))
        url = url_head % j
        Soup = get_soup(url)

        try:
            for a in Soup.find_all("div", {"class": "image"}):
                print(i)
                feature = {}

                link = a.find("a").get("href")
                link = "http://wear.jp" + link
                if type(link) == str:
                    feature = get_feature(link)
                    features[i] = feature
                else:
                    features[i] = None

                img_link = a.find("img").get("data-original")
                img_link = "http:" + img_link
                download_img(img_link, '%d.jpg' % i)
                with open(file_name, mode='wb') as f:
                    pickle.dump(features, f)
                i += 1
                time.sleep(1)
        except requests.exceptions.ConnectionError:
            print("Connection Error")
            time.sleep(5)


