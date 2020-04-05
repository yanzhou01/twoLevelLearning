import numpy as np
import pandas as pd
import re
import MeCab
from tqdm import tqdm_notebook as tqdm
from os import path
import pickle

## 1.　データの読み込み
path = "G:\我的云端硬盘\\fashion_research\data\labels.pkl"

with open(path, 'rb') as handle:
    temp = pickle.load(handle)

labels = pd.DataFrame(temp)

## choose top and bottom features and get one-hot code
### men from 9940.jpg
men_selected = [
    "トップス",
    "トップスcolor",
    "パンツ",
    "パンツcolor"
]
men_labels = labels.loc[9940:, men_selected]
men_labels = men_labels.iloc[:, [0,1,3,4]]

docs = np.array(men_labels)
## 3. 辞書作り of each Question
word2num = [dict(), dict(), dict(), dict()]
num2word = [dict(), dict(), dict(), dict()]

count = [0,0,0,0]
for d in docs:
    for i,w in enumerate(d):
        if w not in word2num[i].keys():
            word2num[i][w] = count[i]
            num2word[i][count[i]] = w
            count[i] += 1
#print(word2num)

## 4.辞書でテキストを数字に変換
ndocs = [[word2num[i][w] for i,w in enumerate(d) ] for d in docs]
print(ndocs[:5])


###　前処理完了

## 4.LDAモデル

def my_LDA(docs, K=5, Iter = 1000, alpha = 0.1, beta = 0.1, trace=False, inter = 10):
    
    def sampling(j, ndk,nkv,nd,nk,i,v):
        probs = np.zeros(K)
        for k in range(K): #<-トピックの数分
            theta = (ndk[i,k]+alpha) / (nd[i] + alpha * M)
            phi  = (nkv[j][k,v]+beta ) / (nk[j][k] + beta * K )
            prob = theta * phi     #<-　トピックi番目の事後確率を計算
            probs[k] = prob #<-　保存
        probs /= probs.sum() #<-　全トピックの確率の和は1にならないので、ここで標準化 
        return np.where(np.random.multinomial(1,probs)==1)[0][0]
    np.random.seed(1000)

    V = [len(J) for J in word2num]  #<-  単語の数
    M = len(docs)      #<-　文章の数

    ndk = np.zeros((M,K)) #<-　文章のトピック分布
    nkv = [np.zeros((K,V[i]))for i in range(len(V))] #<-　トピックの単語分布

    ## 1. すべての単語にランダムに初期値を与える
    topics = [[np.random.randint(K) for w in d] for d in docs]
    for i,d in enumerate(topics):
        for j,z in enumerate(d):  #<- w_d,iの潜在トピックz
            ndk[i,z] += 1 ## z即使话题的值（比如话题1），也是位置索引
            nkv[j][z,ndocs[i][j]] += 1 #话题——单词
    nd = ndk.sum(axis=1) #sum #topics of each doc
    nk = [nkv[i].sum(axis=1) for i in range(len(nkv))]#sum #words of each topic
    if trace:
        chain = []

    for ite in tqdm(range(Iter)):  # Every Iteration
        move = 0
        for i,d in enumerate(topics): # Every Documents
            for j,k in enumerate(d):  # Every word and topics
                ##　事前にサンプリング中の単語を集計から抜く
                v = ndocs[i][j]
                ndk[i,k] -= 1   ## 文档i的全话题计数 -1
                nkv[j][k,v] -= 1   ## 话题单词v的计数 -1
                nk[j][k] -= 1  ## 话题合计单词的计数-1

                ## サンプリング
                new_z = sampling(j, ndk,nkv,nd,nk,i,v)
                
                if trace and ite % inter == 0:  ##　サンプリングで変動があったら動きに＋１
                    if new_z != k:
                        move += 1
                ##　新たな結果の元に、再集計
                topics[i][j] = new_z
                ndk[i,new_z]  += 1 #把抽样的结果赋到相应的话题上
                nkv[j][new_z, v] += 1
                nk[j][new_z]     += 1
        if ite % inter == 0:
            chain.append(move)
    save = {"topics":topics,"nkv":nkv,"ndk":ndk,"nk":nk,"nd":nd}
    if trace:
        save["trace"] = chain
    return save
        

        
def topwords(nkv,t=10):
    sphi = [np.argsort(nkv[i],axis=1).T[::-1].tolist() for i in range(len(nkv))] #[::-1] means reverse [start:stop:tep]
    topwords = [[[num2word[j][i] for i in w] for w in sphi[j]] for j in range(len(sphi))]
    topwords_by_topics = [pd.DataFrame([pd.DataFrame(topwords[i]).iloc[:,z] for i in range(len(topwords))], index = ["Top", "Top_Col", "Bottom", "Bottom_Col"]) for z in range(K)]
    return [topwords_by_topics[i].iloc[:,:5] for i in range(len(topwords_by_topics))]
                


result = my_LDA(ndocs, K=5,trace=True)
print("After sampling\n",result["topics"][:10])

print("Top words\n")
tbt = topwords(result["nkv"])
for i, t in enumerate(tbt):
    print("\n"+f"topic_{i}\n")
    print(t.T)
mem = [t.T for i, t in enumerate(tbt)]
index = [f"topic_{i}" for i in range(K)]
memberships = pd.DataFrame(mem, index=index)
    
#pickle.dump(result, open("result_GoM15.pkl", "wb"))

with open("/Users/zhouyan/Desktop/codes/result_GoM15.pkl", "rb") as pkl:
    result = pickle.load(pkl)