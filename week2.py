from operator import ilshift
import pandas as pd
import numpy as np
import jieba
import copy
import matplotlib.pyplot as plt
from wordcloud import ImageColorGenerator,WordCloud
from PIL import Image

# 文本分词,统计词频
def cut(s):
    freq={}
    words=[]
    for i in range(len(s)):
        #文本分词，以列表形式返回
        s[i] = jieba.lcut(s[i])
        if s[i] == []:
            continue
        cut_list=[]
        for word in s[i]:
            #去除停用词
            if word not in stopwords and word != ' ':
                cut_list.append(word)
                count=freq.get(word,0)
                freq[word]=count+1   
        words.append(cut_list)         
    return words,freq

# 特征词筛选,保留高频词
def search(num,freq):
    high_freq = set()
    for key,value in freq.items():
        if value > num:
            high_freq.add(key)
    return list(high_freq)

# 特征集向量表示(one-hot)
def vector(words,freq):
    vector_array = np.array([[0 for i in range(len(freq))] for j in range(len(words))])
    for i in range(len(words)):
        for j in range(len(freq)):
            if freq[j] in words[i]:
                vector_array[i][j] = 1
    return vector_array

# 计算不同弹幕间的语义相似度(欧几里得距离)
def distance(vector,random_num):
    # 生成随机索引序列
    index1 = np.random.randint(0,len(vector)/2,random_num)
    index2 = np.random.randint(len(vector)/2,len(vector),random_num)
    dist_list = []
    for i in range(len(index1)):
        dist = np.linalg.norm(vector[index1[i]] - vector[index2[i]])
        dist_list.append(dist)
    # 语义相似度最高的弹幕索引
    max_index1 = index1[dist_list.index(max(dist_list))]
    max_index2 = index2[dist_list.index(max(dist_list))]
    return max_index1,max_index2

# 根据特征向量矩阵计算重心，进而计算代表性评论
def center(matrix):
    # 确定重心
    count = [0] * len(matrix[0])
    for row in range(0, len(matrix)):
        for column in range(0, len(matrix[0])):
            count[column] += matrix[row][column]
    for column in range(0, len(count)):
        count[column] = count[column] / len(matrix)
    gravity_center = count
    # 计算各向量与重心的距离,获取最小值索引
    dis_list = []
    for row in range(0, len(matrix)):
        dis = np.sqrt(np.sum(np.square(gravity_center - matrix[row])))
        dis_list.append(dis)
    index = dis_list.index(min(dis_list))
    return index

#可视化词云函数
def word_cloud(high_freq,freq):
    # 建立高频词频数字典
    top_words = {}
    for item in high_freq:
        top_words[item] = freq[item]
    # 配置参数
    mk = np.array(Image.open('D:\python练习\week2\heart.jpg'))
    font = 'C:\Windows\Fonts\STXINGKA.TTF'
    w = WordCloud(mask = mk, width = 626, height = 626, min_font_size = 5,
    max_words = len(high_freq), font_path = font, background_color = 'white')
    #生成词云
    image_colors = ImageColorGenerator(mk)
    w.generate_from_frequencies(top_words)
    plt.imshow(w.recolor(color_func=image_colors))
    plt.axis('off')
    plt.show()
    return

# 主函数
# 读入弹幕数据
danmuku = pd.read_csv('D:\python练习\week2\danmuku.csv')
danmu = list(danmuku.iloc[:,0])
danmu_deepcocy = copy.deepcopy(danmu)

# 停用词加入自定义词典，建立停用词列表
stopwords = []
jieba.load_userdict("D:\python练习\week2\stopwords_list.txt")
with open('D:\python练习\week2\stopwords_list.txt','r',encoding='utf-8-sig') as f:
    for word in f:
        if len(word)>0:
            stopwords.append(word.strip())

# 文本分词,获得词频字典
danmu_cut,freq_dict = cut(danmu)

# 筛选高频词
search_value = 10000
high_freq = search(search_value,freq_dict)
print('出现频率大于{}的特征集为{}'.format(search_value,high_freq))

# 弹幕向量表示
vector_array = vector(danmu_cut,high_freq)

# 计算语义相似度，获取语义相似度最高的两句弹幕索引
max_index1,max_index2 = distance(vector_array,100000)
print('语义相似度最高的两句弹幕为“{}”和“{}”'.format(danmu_deepcocy[max_index1],danmu_deepcocy[max_index2]))

# 计算代表性弹幕(向量重心)
print('代表性弹幕为：“{}”'.format(danmu_deepcocy[center(vector_array)]))

# 词云可视化高频词
word_cloud(high_freq,freq_dict)