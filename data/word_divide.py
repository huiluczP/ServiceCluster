import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

"""
文档预处理
分句，分词，提取词语主干
"""

# 获取英语的停止词
stop = stopwords.words("english")
# 获取wordNet词形还原帮助类
lemma = WordNetLemmatizer()


def isSymbol(word):
    return bool(re.match(r'[^\w]', word))


def hasNumber(word):
    return bool(re.search(r'\d', word))


def check(word):
    if isSymbol(word):
        return False
    if hasNumber(word):
        return False
    return True


def divide(doc):
    # 对每个文档进行分句，分词，归一化处理
    # doc为单文档字符串
    sentences = nltk.sent_tokenize(doc)
    words = []
    for sentence in sentences:
        word_list = nltk.word_tokenize(sentence)
        for word in word_list:
            word = word.lower()
            if check(word) and not (word in stop):
                # 归一形态
                temp = lemma.lemmatize(word)
                # 获取词根
                lem = wordnet.morphy(word)
                if lem is None:
                    words.append(word)
                else:
                    words.append(lem)
    return words


def get_doc_after_divide(doc):
    # 获取完整处理后的语料信息，返回为dictionary
    # doc为为文档字典，未分词
    dic = {}
    for k in doc:
        document = doc[k]
        words = divide(document)
        doc_word = " ".join(words)
        dic[k] = doc_word
    return dic