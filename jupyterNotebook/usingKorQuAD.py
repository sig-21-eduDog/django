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

# 코드 출처
# https://github.com/kimwoonggon/publicservant_AI/blob/master/05_%EC%BC%80%EB%9D%BC%EC%8A%A4%EB%A1%9C_KorQuAD(%ED%95%9C%EA%B5%AD%EC%96%B4_Q%26A)_%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0.ipynb

# !pip install wget

# + colab={"base_uri": "https://localhost:8080/", "height": 210} id="SwK1tYV_uayz" outputId="d91212f4-510b-4345-ec54-a9f029cea175"
# !wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
# -

# !pip list

# + id="Sx2147Ndq3lS"
import os
if "bert" not in os.listdir():
  os.makedirs("bert")
else:
  pass

import zipfile
import shutil
         
bert_zip = zipfile.ZipFile('multi_cased_L-12_H-768_A-12.zip')
bert_zip.extractall('bert')
 
bert_zip.close()


# + id="3yaskz4Aujj1"
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

copytree("bert/multi_cased_L-12_H-768_A-12", "bert")

# + colab={"base_uri": "https://localhost:8080/", "height": 122} id="zXWE_XTqmta6" outputId="04d88eb1-22b8-4127-babc-afad9b1bdbc0"
os.listdir()

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

pretrained_path ="bert"
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
  print("Question : ", question)
  
  print("-"*50)
  print("Context : ", end = " ")
  
  def split_text(text, n):
    for line in text.splitlines():
        while len(line) > n:
           x, line = line[:n], line[n:]
           yield x
        yield line

  

  for line in split_text(doc, 150):
    print(line)

  print("-"*50)
  print("ANSWER : ", end = " ")
  print("\n")
  sentences = []
  
  for i in range(start, end+1):
    token_based_word = reverse_token_dict[indexes[i]]
    sentences.append(token_based_word)
    print(token_based_word, end= " ")
  
  print("\n")
  print("Untokenized Answer : ", end = "")
  for w in sentences:
    if w.startswith("##"):
      w = w.replace("##", "")
    else:
      w = " " + w
    
    print(w, end="")
  print("")
  print(start, end)

# + id="T_I3QsA99ZhH"
doc = "2000년 5월 소리바다에서 개발한 mp3 음악파일 교환 서비스로 P2P 프로그램으로 출발해 대표적인 음악파일 교환 서비스로 성장하였다. 저작권 침해 문제로 서비스 중지 가처분 결정을 받았으나 소리바다5로 2006년 7월 전면 유료화 서비스로 전환하였고 월정액제와 자유이용권 등의 제도를 운영하고 있다.사용자끼리 서로의 mp3 파일을 검색하고다운로드받을 수 있는 개인 대 개인(P2P) 프로그램으로 출발하였다. 2000년 5월부터 온라인 서비스에 들어가 국내 대표적인음악파일 교환 서비스로 자리잡았다. 그러나 한국음반산업협회에서 소리바다를 상대로가처분신청을 제기하였고, 2002년 7월 11일법원은 소리바다의 음악파일 공유 서비스가저작권을 침해할 소지가 있다고 판단해 서비스 중지 가처분 결정을 내려 7월 31일 이후 서비스가 중단됐다.상황이 이렇게 되자 8월 24일 중앙집중식 검색 기능을 없앤 새 파일 교환 프로그램 소리바다 2를 개발하였다. 이 프로그램은 메인 서버 없이 슈퍼피어(SuperPeer) 방식으로 사용자 리스트를 받을 수 있으며, 전반적으로 사용자 인터페이스를 개선한 것이었다.소리바다는 2003년 11월 주식회사로 법인 전환을 하였다. 2004년 7월 '소리바다3'를 출시하였으며, 12월 유료 mp3＃을 열었다. 2005년 11월 서비스를 중단하였다가 2006년 3월 '소리바다5'라는 이름으로 서비스를 재개하였으며, 7월부터 전면 유료화되었다. '소리바다5'는 회원들이 공유한 mp3파일을 실시간으로 검색하여 원하는 파일을 다운로드할 수 있는 P2P 프로그램으로 운영되며, 월정액제의 자유이용권을 구매하여 이용할 수 있다. 다운로드하면서 미리듣기를 할 수 있고, 원하는 mp3파일의 가사도 볼 수 있다. 또 오르골이라는 무제한 용량의 저장공간도 제공된다."
question = "mp3는 언제 개발 되었어?"

predict_letter(question, doc)

# +
doc = "태양계의 5번째 행성이며, 태양계의 행성 중 가장 부피가 크고 무겁다. 반지름은 지구의 11.2배, 부피는 지구의 1300배가 넘는다. 질량은 지구의 318배다. 부피에 비해 질량이 작은 이유는 암석형 행성보다 밀도가 낮은 성분들이 주요 구성성분인 가스형 행성이기 때문이다. 그럼에도 목성의 질량은 다른 태양계 행성들을 합친 것보다도 무겁다. 심지어 그 7개 행성의 질량을 몽땅 다 합쳐도 목성의 절반도 되지 않는다. 태양계에서 태양이 99.86%를 차지하고, 목성은 나머지 0.14% 중에서 약 2/3인 0.095%를 차지한다. 뒤를 이어 토성이 0.029%를 차지하며, 나머지 행성들을 모두 합쳐도 태양계 질량의 0.016% 정도 밖에 되지 않는다."
question = "목성의 부피는 지구의 몇배인가?"

predict_letter(question, doc)

# + colab={"base_uri": "https://localhost:8080/", "height": 283} id="oQFOPwW-WsAN" outputId="d2e63a4c-c1bf-47ae-bb86-6f2354f50cb2"

doc = "사스와 메르스처럼 코로나 바이러스의 보고되지 않은 종에 인한 감염으로 발생하는 호흡기 질환이다. 최초 발생 원인과 전파 경로는 아직 정확히 밝혀지지 않았다.발병 초기에 보고된 불상의 폐병 증상으로 대중적으로는 '우한 폐렴' 등의 키워드로 불렸으며, 현재 세계보건기구(WHO)에서는 Novel coronavirus(2019-nCoV)라는 표현을 사용 중이다(#). 미국, 영국 등 일부 외신에서는 Wuhan coronavirus라는 표현을 사용하기도 한다. 현재 미국 정부는 질병 명칭에는 Wuhan을 넣지 않은 Coronavirus라고만 지칭 중이다.[11] 대한민국 정부에서 잠정적으로 사용하는 질병 명칭은 신종 코로나바이러스감염증이다. 이는 2015년부터 낙인 효과를 우려한 WHO에서 병명에 지역 이름을 넣는 것을 피하도록 한 권고[12](#)에 따른 것이다. 물론 해당 권고는 구속력이 없으며, 정부나 공공 의료기관이 아닌 일반인, 그리고 언론이 우한 폐렴을 사용하는 것에도 문제가 없다. 실제로 외신이나 당사자인 중국 본토에서도 '우한' 혹은 '중국' 명칭을 사용하고 있다. #기사 사태 초반에는 우한시 안에 국한될 것으로 판단하는 이들이 많았으나, 점차 우한 외 후베이성과 인근 지역, 그리고 중국을 벗어나 해외로까지 퍼지면서 상황이 심각해졌다. 우한시의 인구는 약 1,100만 명이며, 하필 맞물린 춘절[13]로 인한 인구 대이동으로 병이 사방으로 퍼질 위험이 있어 중국 현지에 비상이 걸렸다. 게다가 춘절 기간 중 중국인 관광객들이 해외여행을 많이 가다보니 세계 여러 나라에서도 촉각이 곤두서있다. 따라서 춘절 전후가 감염병 확산의 고비라 할 수 있다."
question = "우한 폐렴은 어디서 발생하였는가?"

predict_letter(question, doc)

# +
question = "MP4의 단점이 뭐야"
doc = "MP4PMP3 플레이어에 동영상을 볼 수 있는 기능을 추가한 휴대형멀티미디어재생기. 고효율 압축 기술인 MPEG 4(H.264)와 고성능 칩 셋, 그리고 컬러 LCD 기술 개발로 실용화가 가능해졌다. MP4 플레이어는 MP3처럼 누구나 쉽게 사용할 수 있으며, 다양한 부가 기능이 추가된 것이 특징이다. 기존 어학용 및 교육용으로 MP3 플레이어를 사용했던 소비자들은 MP4 플레이어로 실제 강의 화면을 보면서 학습을 할 수 있기 때문에 MP3보다 활용도가 뛰어나다. 다만 동영상을 재생하기 때문에 MP3 플레이어에 비해 배터리 소모가 많고, PMP에 비해 화면이 작고 화질이 떨어지는 단점이 있다"

predict_letter(question, doc)
# -

question = "자연어 처리에서 1990년대 이후 무엇을 활용했어?"
doc = "컴퓨터를 이용해 사람의 자연어를 분석하고 처리하는 기술.약어NLP요소 기술로 자연어 분석, 이해, 생성 등이 있으며,정보 검색,기계 번역, 질의응답 등 다양한 분야에 응용된다.자연어는 일반 사회에서 자연히 발생하여 사람이 의사소통에 사용하는 언어로, 컴퓨터에서 사용하는 프로그래밍 언어와 같이 사람이 의도적으로 만든 인공어(constructed language)에 대비되는 개념이다.자연어 처리에는 자연어 분석, 자연어 이해, 자연어 생성 등의 기술이 사용된다. 자연어 분석은 그 정도에 따라 형태소 분석(morphological analysis), 통사 분석(syntactic analysis),의미 분석(semantic analysis) 및 화용 분석(pragmatic analysis)의 4 가지로 나눌 수 있다. 자연어 이해는 컴퓨터가 자연어로 주어진 입력에 따라 동작하게 하는 기술이며, 자연어 생성은 동영상이나 표의 내용 등을 사람이 이해할 수 있는 자연어로 변환하는 기술이다.자연어 처리는인공 지능의 주요 분야 중 하나로, 1950년대부터 기계 번역과 같은 자연어 처리 기술이 연구되기 시작했다. 1990년대 이후에는 대량의 말뭉치(corpus) 데이터를 활용하는기계 학습기반 및 통계적 자연어 처리 기법이 주류가 되었으며, 최근에는심층 기계 학습(deep learning) 기술이 기계 번역 및 자연어 생성 등에 적용되고 있다.※ 통사 분석(syntactic anaylisys): 컴퓨터 분야에서는 ‘구문 해석’이라 하나 언어학에서는 ‘통사 분석’이라 함. 통사는 생각이나 감정을 말과 글로 표현할 때 완결된 내용을 나타내는 최소의 단위.※ 통사: 주어와 서술어를 갖추고 있는 것이 원칙이나 때로 이런 것이 생략될 수도 있다. 글의 경우, 문장의 끝에 ‘.’, ‘?’, ‘!’ 따위의 마침표를 찍는다. ‘철수는 몇 살이니?’, ‘세 살.’, ‘정말?’ 따위이다. (출처: 우리말샘)※ 화용(話用) 분석: 말하는 이, 듣는 이, 시간, 장소 따위로 구성되는 맥락과 관련하여 문장의 의미를 체계적으로 분석하는 것"
predict_letter(question, doc)

question = "자연어 처리에서 1950년대 부터 무엇을 연구했어?"
doc = "컴퓨터를 이용해 사람의 자연어를 분석하고 처리하는 기술.약어NLP요소 기술로 자연어 분석, 이해, 생성 등이 있으며,정보 검색,기계 번역, 질의응답 등 다양한 분야에 응용된다.자연어는 일반 사회에서 자연히 발생하여 사람이 의사소통에 사용하는 언어로, 컴퓨터에서 사용하는 프로그래밍 언어와 같이 사람이 의도적으로 만든 인공어(constructed language)에 대비되는 개념이다.자연어 처리에는 자연어 분석, 자연어 이해, 자연어 생성 등의 기술이 사용된다. 자연어 분석은 그 정도에 따라 형태소 분석(morphological analysis), 통사 분석(syntactic analysis),의미 분석(semantic analysis) 및 화용 분석(pragmatic analysis)의 4 가지로 나눌 수 있다. 자연어 이해는 컴퓨터가 자연어로 주어진 입력에 따라 동작하게 하는 기술이며, 자연어 생성은 동영상이나 표의 내용 등을 사람이 이해할 수 있는 자연어로 변환하는 기술이다.자연어 처리는인공 지능의 주요 분야 중 하나로, 1950년대부터 기계 번역과 같은 자연어 처리 기술이 연구되기 시작했다. 1990년대 이후에는 대량의 말뭉치(corpus) 데이터를 활용하는기계 학습기반 및 통계적 자연어 처리 기법이 주류가 되었으며, 최근에는심층 기계 학습(deep learning) 기술이 기계 번역 및 자연어 생성 등에 적용되고 있다.※ 통사 분석(syntactic anaylisys): 컴퓨터 분야에서는 ‘구문 해석’이라 하나 언어학에서는 ‘통사 분석’이라 함. 통사는 생각이나 감정을 말과 글로 표현할 때 완결된 내용을 나타내는 최소의 단위.※ 통사: 주어와 서술어를 갖추고 있는 것이 원칙이나 때로 이런 것이 생략될 수도 있다. 글의 경우, 문장의 끝에 ‘.’, ‘?’, ‘!’ 따위의 마침표를 찍는다. ‘철수는 몇 살이니?’, ‘세 살.’, ‘정말?’ 따위이다. (출처: 우리말샘)※ 화용(話用) 분석: 말하는 이, 듣는 이, 시간, 장소 따위로 구성되는 맥락과 관련하여 문장의 의미를 체계적으로 분석하는 것"
predict_letter(question, doc)


