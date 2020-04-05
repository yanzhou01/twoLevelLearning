import numpy as np
import pandas as pd
import re
import MeCab
from tqdm import tqdm_notebook as tqdm
from os import path

## 1.　データの読み込み
# comments = pd.read_csv("/Users/zhouyan/Desktop/codes/iphone7.csv",encoding="cp932")
# print(comments.head())

file_name = '/Volumes/GoogleDrive/我的云端硬盘/fashion_research/data/features.pickle'
with open(file_name, 'rb') as handle:
    tempDictFeatures = pickle.load(handle)

data = pd.DataFrame.from_dict(tempDictFeatures, orient = 'index')
#data.to_excel("rawData.xlsx")

women = data[:9940]
men = data[9940:]
men_text = []
men_des = men.description.values

comments = pd.DataFrame(men_des, columns=["comment"])

## 2.　日本語形態素解析
def get_words(text,conditions):
    t = MeCab.Tagger()
    text = re.subn("\W","",text)[0]   #<-　追加、記号抜き
    tmp = t.parse(text)
    tmp = [line.split("\t") for line in tmp.split("\n")][:-2]
    words = []
    for i in tmp:
        if i[1].split(",")[0] in conditions:
            words.append(i[0])
    return words

docs = [get_words(text,["動詞","形容詞","名詞"]) for text in comments.iloc[:,0]]
print("形態素解析完了")
print(docs[:5])  # <-最初の5人のコメント

## 3. 辞書作り
word2num = dict()
num2word = dict()

count = 0
for d in docs:
    for w in d:
        if w not in word2num.keys():
            word2num[w] = count
            num2word[count] = w
            count += 1
#print(word2num)

## 4.辞書でテキストを数字に変換
ndocs = [[word2num[w] for w in d ] for d in docs]
print(ndocs[:5])


###　前処理完了

## 4.LDAモデル

def my_LDA(docs, K=5, Iter = 1000, alpha = 0.1, beta = 0.1, trace=False, inter = 10):
    
    def sampling(ndk,nkv,nd,nk,i,v):
        probs = np.zeros(K)
        for k in range(K): #<-トピックの数分
            theta = (ndk[i,k]+alpha) / (nd[i] + alpha * M)
            phi  = (nkv[k,v]+beta ) / (nk[k] + beta * K )
            prob = theta * phi     #<-　トピックi番目の事後確率を計算
            probs[k] = prob #<-　保存
        probs /= probs.sum() #<-　全トピックの確率の和は1にならないので、ここで標準化 
        # 取得结果为1的位置 （array[z]，）[0][0] = z
        return np.where(np.random.multinomial(1,probs)==1)[0][0]
    np.random.seed(1000)

    V = len(word2num)  #<-  単語の数
    M = len(docs)      #<-　文章の数

    ndk = np.zeros((M,K)) #<-　文章のトピック分布
    nkv = np.zeros((K,V)) #<-　トピックの単語分布

    ## 1. すべての単語にランダムに初期値を与える
    topics = [[np.random.randint(K) for w in d] for d in docs]
    for i,d in enumerate(topics):
        for j,z in enumerate(d):  #<- w_d,iの潜在トピックz
            ndk[i,z] += 1 ## z即使话题的值（比如话题1），也是位置索引
            nkv[z,ndocs[i][j]] += 1 #话题——单词 #n?
    nd = ndk.sum(axis=1) #sum #topics of each doc
    nk = nkv.sum(axis=1) #sum #words of each topic
    if trace:
        chain = []

    for ite in tqdm(range(Iter)):  # Every Iteration
        move = 0
        for i,d in enumerate(topics): # Every Documents
            for j,k in enumerate(d):  # Every word and topics
                ##　事前にサンプリング中の単語を集計から抜く
                v = ndocs[i][j]
                ndk[i,k] -= 1   ## 文档i的全话题计数 -1
                nkv[k,v] -= 1   ## 话题单词v的计数 -1
                nk[k] -= 1  ## 话题合计单词的计数-1

                ## サンプリング
                new_z = sampling(ndk,nkv,nd,nk,i,v)
                
                if trace and ite % inter == 0:  ##　サンプリングで変動があったら動きに＋１
                    if new_z != k:
                        move += 1
                ##　新たな結果の元に、再集計
                topics[i][j] = new_z
                ndk[i,new_z]  += 1 #把抽样的结果赋到相应的话题上
                nkv[new_z, v] += 1
                nk[new_z]     += 1
        if ite % inter == 0:
            chain.append(move)
    save = {"topics":topics,"nkv":nkv,"ndk":ndk,"nk":nk,"nd":nd}
    if trace:
        save["trace"] = chain
    return save
        

        
def topwords(nkv,t=10):
    sphi = np.argsort(nkv,axis=1).T[::-1].tolist()
    topwords = [[num2word[i] for i in w] for w in sphi]
    return pd.DataFrame(topwords).iloc[:10,:]
                


result = my_LDA(ndocs, K=15,trace=True)
print("After sampling\n",result["topics"][:10])
print("Top words\n",topwords(result["nkv"]))

# with open("/Users/zhouyan/Desktop/codes/result_LDA15.pkl", "rb") as pkl:
#     result = pickle.load(pkl)