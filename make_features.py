import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

def removeHTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)

# 不要な文字列の削除
def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", '',x)
    x = re.sub("'\d+", '',x)
    x = re.sub("\d+", '',x)
    x = re.sub("http\w+", '',x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x

# テキストに対してTfidfVectorizerの特徴量を作成
def apply_tfidf(train, test, text_columns):
    # 'review' 用
    vectorizer1 = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 2),
        min_df=0.02,
        max_df=0.85,
        sublinear_tf=True,
    )
    # 'replyContent' 用 
    vectorizer2 = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0.02,
        max_df=0.85,
        sublinear_tf=True,
    )

    for idx, col in enumerate(text_columns):
        vectorizer = vectorizer1 if idx == 0 else vectorizer2
        
        train_tfid = vectorizer.fit_transform(tqdm([i for i in train[col]], desc=f"TfidfVectorizer {col}"))
        test_tfid = vectorizer.transform(tqdm([i for i in test[col]], desc=f"TfidfVectorizer {col}"))

        train_dense_matrix = train_tfid.toarray()
        test_dense_matrix = test_tfid.toarray()

        train_tmp = pd.DataFrame(train_dense_matrix)
        test_tmp = pd.DataFrame(test_dense_matrix)

        train_tfid_columns = [f'tfid_{col}_{i}' for i in range(len(train_tmp.columns))]
        test_tfid_columns = [f'tfid_{col}_{i}' for i in range(len(test_tmp.columns))]

        train_tmp.columns = train_tfid_columns
        test_tmp.columns = test_tfid_columns

        train = pd.concat([train.reset_index(drop=True), train_tmp.reset_index(drop=True)], axis=1)
        test = pd.concat([test.reset_index(drop=True), test_tmp.reset_index(drop=True)], axis=1)

    return train, test

# テキストに対してCountVectorizerの特徴量を作成
def apply_cv(train, test, text_columns):
    # 'review' 用
    vectorizer1 = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 2),
        min_df=0.02,
        max_df=0.85,
    )
    # 'replyContent' 用 
    vectorizer2 = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 3),
        min_df=0.02,
        max_df=0.85,
    )

    for idx, col in enumerate(text_columns):
        vectorizer = vectorizer1 if idx == 0 else vectorizer2
        
        train_tfid = vectorizer.fit_transform(tqdm([i for i in train[col]], desc=f"CountVectorizer {col}"))
        test_tfid = vectorizer.transform(tqdm([i for i in test[col]], desc=f"CountVectorizer {col}"))

        train_dense_matrix = train_tfid.toarray()
        test_dense_matrix = test_tfid.toarray()

        train_tmp = pd.DataFrame(train_dense_matrix)
        test_tmp = pd.DataFrame(test_dense_matrix)

        train_tfid_columns = [f'cntvec_{col}_{i}' for i in range(len(train_tmp.columns))]
        test_tfid_columns = [f'cntvec_{col}_{i}' for i in range(len(test_tmp.columns))]

        train_tmp.columns = train_tfid_columns
        test_tmp.columns = test_tfid_columns

        train = pd.concat([train.reset_index(drop=True), train_tmp.reset_index(drop=True)], axis=1)
        test = pd.concat([test.reset_index(drop=True), test_tmp.reset_index(drop=True)], axis=1)

    return train, test

cList = {
    "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have",
    "couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not",
    "hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
    "he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is",
    "I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","isn't": "is not",
    "it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam",
    "mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
    "mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
    "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have",
    "that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would",
    "they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
    "what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
    "where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
    "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
    "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you will",
    "you'll've": "you will have","you're": "you are","you've": "you have"
}

# cListを用いてテキストを統一
def expand_contractions(text, cList):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in cList.keys()) + r')\b')
    def replace(match):
        return cList[match.group(0)]
    return pattern.sub(replace, text)

def prepro(df, vectorizer=None):
    df['new_review'] = [dataPreprocessing(x) for x in df['review']]
    df['new_replyContent'] = [dataPreprocessing(x) for x in df['replyContent']]
    
    df['new_review'] = df['new_review'].apply(lambda x: expand_contractions(x, cList))
    df['new_replyContent'] = df['new_replyContent'].apply(lambda x: expand_contractions(x, cList))
    
    df['new_review'] = df.apply(lambda row: f"the version of the app is {row['reviewCreatedVersion']}. {row['new_review']}" if 'reviewCreatedVersion' in df.columns else row['new_review'], axis=1)
    df['new_review'] = df.apply(lambda row: f"the number of likes for this review is {row['thumbsUpCount']}. {row['new_review']}" if 'thumbsUpCount' in df.columns else row['new_review'], axis=1)
    
    df['new_replyContent'] = df.apply(lambda row: f"it took {row['timeToReply']} for the developer to reply to a user's review. {row['new_replyContent']}" if 'timeToReply' in df.columns else row['new_replyContent'], axis=1)
    
    return df

train = prepro(train)
test = prepro(test)

train.info()
print(train.head(10))
test.info()
print(test.head(10))

# 2つのテキストに対して特徴量を作成
text_columns = ['new_review', 'new_replyContent']
train, test = apply_tfidf(train, test, text_columns)
train, test = apply_cv(train, test, text_columns)

tfid_columns = [col for col in train.columns if "tfid" in col]
cntvec_columns = [col for col in train.columns if "cntvec" in col]

# 新しいDataFrameを作成
train_tfid = train[tfid_columns]
train_cntvec = train[cntvec_columns]
test_tfid = test[tfid_columns]
test_cntvec = test[cntvec_columns]

# 新しいCSVファイルに保存
train_tfid.to_csv('datasets/new_train_tfid_features.csv', index=False)
train_cntvec.to_csv('datasets/new_train_cntvec_features.csv', index=False)
test_tfid.to_csv('datasets/new_test_tfid_features.csv', index=False)
test_cntvec.to_csv('datasets/new_test_cntvec_features.csv', index=False)