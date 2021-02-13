# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: RLyoJulGae
#     language: python
#     name: rljg
# ---


# + colab={"base_uri": "https://localhost:8080/", "height": 81} id="Ffa---Ov8WfF" outputId="6e28b257-6558-4eec-8015-7bd1b398fb8b"
import tensorflow as tf

import pandas as pd
import numpy as np  
import re
import pickle

import keras as keras
from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers

from keras import backend as K
from keras.layers import Layer

import codecs
from tqdm import tqdm
import shutil
import json

import os

# + id="bniU2r9PouUO"
import warnings
import tensorflow as tf
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

# + [markdown] id="A3bWhmOP0mOv"
# 케라스에서 Bert 활용을 쉽게 만들어주는 모듈 keras-bert를 설치합니다<br>그리고 Adam optimizer의 수정판인 keras-radam 모듈을 임포트합니다.

# + id="FZv-5ALnGdwI"
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps

from keras_radam import RAdam


# + [markdown] id="kVGK2j5tKJgS"
# SQUAD JSON파일을 PANDAS DATAFRAME으로 만들어주는 함수를 정의합니다. 
#
# KorQuAD도 SQUAD랑 동일한 방식이기에, Pandas Dataframe으로 잘 변환 됩니다. 
# 출처 : https://www.kaggle.com/sanjay11100/squad-stanford-q-a-json-to-pandas-dataframe

# + id="zgWmFtiaIZQK"
def squad_json_to_dataframe_train(input_file_path, record_path = ['data','paragraphs','qas','answers'], verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


# + [markdown] id="Eoez6gHVGeP8"
# KorQUAD 데이터를 PANDAS DATAFRAME 형식으로 로드합니다.

# + [markdown] id="2-NebUQmG1w-"
#
# - bert 훈련을 위한 사전 설정을 합니다. SEQ_LEN은 문장의 최대 길이입니다. SEQ_LEN 보다 문장의 길이가 작다면 남은 부분은 0이 채워지고, 만약에 SEQ_LEN보다 문장 길이가 길다면 SEQ_LEN을 초과하는 부분이 잘리게 됩니다.  
# 본 문제에서는 메모리 문제 등으로 384로 정했습니다.
# - BATCH_SIZE는 메모리 초과 같은 문제를 방지하기 위해 작은 수인 10으로 정했습니다. 그리고 총 훈련 에포크 수는 1로 정했습니다. 학습율(LR;Learning rate)은 3e-5로 작게 정했습니다.
# - pretrained_path는 bert 사전학습 모형이 있는 폴더를 의미합니다.
# - 그리고 우리가 분석할 문장이 들어있는 칼럼의 제목인 document와 긍정인지 부정인지 알려주는 칼럼을 label로 정해줍니다
#

# + id="tLYWgZJAtsku"
SEQ_LEN = 384
BATCH_SIZE = 3
EPOCHS=1
LR=3e-5

pretrained_path = "bert"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

DATA_COLUMN = "context"
QUESTION_COLUMN = "question"
TEXT = "text"

# + [markdown] id="IaU-13a9H4YJ"
# vocab.txt에 있는 단어에 인덱스를 추가해주는 token_dict라는 딕셔너리를 생성합니다.  
# 우리가 분석할 문장이 토큰화가 되고, 그 다음에는 인덱스(숫자)로 변경되어서 버트 신경망에 인풋으로 들어게 됩니다.

# + id="-ph2YQXg2iz5"
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        if "_" in token:
          token = token.replace("_","")
          token = "##" + token
        token_dict[token] = len(token_dict)


# + [markdown] id="0WhiwaBCIXqv"
# - BERT의 토큰화는 단어를 분리하는 토큰화 방식입니다. wordpiece(단어조각?) 방식이라고 하는데, 이는 한국어를 형태소로 꼭 변환해야 할 문제를 해결해주며, 의미가 있는 단어는 밀접하게 연관이 되게 하는 장점까지 갖추고 있습니다.
# - 단어의 첫 시작은 ##가 붙지 않지만, 단어에 포함되면서 단어의 시작이 아닌 부분에는 ##가 붙는 것이 특징입니다.  
# - 네이버 감성분석에서 했던 것처럼, 한국어 감성분석에서는 새로 토크나이저 클래스를 상속을 받아서, 토크나이저를 재정의 해주어야 합니다.(그렇지 않으면 완전자모분리 현상 발생)

# + id="pxI9s2TeAYuz"
class inherit_Tokenizer(Tokenizer):
  def _tokenize(self, text):
        if not self._cased:
            text = text
            
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens


# + id="88T_cMA4tsk2"
tokenizer = inherit_Tokenizer(token_dict)

# + [markdown] id="W0dmxrOhJ9vD"
# 토큰화가 잘 되었는지 확인해 봅니다.
# 버트 모형은 문장 앞에 꼭 [CLS]라는 문자가 위치하고, [SEP]라는 문자가 끝에 위치합니다.  
# [CLS]는 문장의 시작, [SEP]는 문장의 끝을 의미합니다.

# + id="oF_7BhoWw6_E"
reverse_token_dict = {v : k for k, v in token_dict.items()}

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="8auSqxdHtslD" outputId="703519f8-ab7c-4a47-b1b2-65215b464012"
layer_num = 12
model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=False,
    trainable=True,
    seq_len=SEQ_LEN,)
model.summary()


# + id="H99O5l4X8xyb"
class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape  


# + id="gOBsah1UgIPq"
class MyLayer_Start(Layer):

    def __init__(self,seq_len, **kwargs):
        
        self.seq_len = seq_len
        self.supports_masking = True
        super(MyLayer_Start, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[2],2),
                                 initializer='uniform',
                                 trainable=True)
        super(MyLayer_Start, self).build(input_shape)

    def call(self, x):
        
        x = K.reshape(x, shape=(-1,self.seq_len,K.shape(x)[2]))
        x = K.dot(x, self.W)
        
        x = K.permute_dimensions(x, (2,0,1))

        self.start_logits, self.end_logits = x[0], x[1]
        
        self.start_logits = K.softmax(self.start_logits, axis=-1)
        
        return self.start_logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len)


class MyLayer_End(Layer):
  def __init__(self,seq_len, **kwargs):
        
        self.seq_len = seq_len
        self.supports_masking = True
        super(MyLayer_End, self).__init__(**kwargs)
  
  def build(self, input_shape):
        
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[2], 2),
                                 initializer='uniform',
                                 trainable=True)
        super(MyLayer_End, self).build(input_shape)

  
  def call(self, x):

        
        x = K.reshape(x, shape=(-1,self.seq_len,K.shape(x)[2]))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (2,0,1))
        
        self.start_logits, self.end_logits = x[0], x[1]
        
        self.end_logits = K.softmax(self.end_logits, axis=-1)
        
        return self.end_logits

  def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len)

# + [markdown] id="3uYTT6WYgiBG"
# BERT 모델을 출력하는 함수를 지정합니다.  
# start_answer, end_answer를 예측하게 됩니다.

# + id="NcbOzpEb0wxU"
from keras.layers import merge, dot, concatenate
from keras import metrics
def get_bert_finetuning_model(model):
  inputs = model.inputs[:2]
  dense = model.output
  x = NonMasking()(dense)
  outputs_start = MyLayer_Start(SEQ_LEN)(x)
  outputs_end = MyLayer_End(SEQ_LEN)(x)
  bert_model = keras.models.Model(inputs, [outputs_start, outputs_end])
  bert_model.compile(
      optimizer=RAdam(learning_rate=LR, decay=0.0001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  
  return bert_model


# + id="DwMLCYp0TzTv"
bert_model = get_bert_finetuning_model(model)
bert_model.load_weights("korquad_wordpiece_3.h5")


# + [markdown] id="YeXe8E-0i1si"
# Test data set에 대한 bert_input을 만들어 줍니다.  
# Train data set과는 다르게 label을 생성하지 않습니다.

# + id="q_WdnjmMYZ-N"
def convert_pred_data(question, doc):
    global tokenizer
    indices, segments = [], []
    ids, segment = tokenizer.encode(question, doc, max_len=SEQ_LEN)
    indices.append(ids)
    segments.append(segment)
    indices_x = np.array(indices)
    segments = np.array(segments)
    return [indices_x, segments]

def load_pred_data(question, doc):
    data_x = convert_pred_data(question, doc)
    return data_x


# + [markdown] id="T4LmVp-5lOnR"
# 질문과 문장을 받아 답을 알려주는 함수를 정의합니다.

# + id="Engs2SIlW0SP"
def predict_letter(question, doc):
    test_input = load_pred_data(question, doc)
    test_start, test_end = bert_model.predict(test_input)

    indexes = tokenizer.encode(question, doc, max_len=SEQ_LEN)[0]
    start = np.argmax(test_start, axis=1).item()
    end = np.argmax(test_end, axis=1).item()
    start_tok = indexes[start]
    end_tok = indexes[end]

    def split_text(text, n):
        for line in text.splitlines():
            while len(line) > n:
                x, line = line[:n], line[n:]
                yield x
            yield line

    sentences = []
    for i in range(start, end + 1):
        token_based_word = reverse_token_dict[indexes[i]]
        sentences.append(token_based_word)

    word = ''
    for w in sentences:
        if w.startswith("##"):
            w = w.replace("##", "")
        else:
            w = " " + w
        word = word + w

    if word[0] == " ":
        word = word[1:]

    return word
