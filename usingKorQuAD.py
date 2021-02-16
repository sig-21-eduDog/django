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

# + id="tLYWgZJAtsku"
SEQ_LEN = 384
BATCH_SIZE = 3
EPOCHS=1
LR=3e-5

pretrained_path = "bert2"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

DATA_COLUMN = "context"
QUESTION_COLUMN = "question"
TEXT = "text"


# + [markdown] id="0WhiwaBCIXqv"
# - BERT의 토큰화는 단어를 분리하는 토큰화 방식입니다. wordpiece(단어조각?) 방식이라고 하는데, 이는 한국어를 형태소로 꼭 변환해야 할 문제를 해결해주며, 의미가 있는 단어는 밀접하게 연관이 되게 하는 장점까지 갖추고 있습니다.
# - 단어의 첫 시작은 ##가 붙지 않지만, 단어에 포함되면서 단어의 시작이 아닌 부분에는 ##가 붙는 것이 특징입니다.  
# - 네이버 감성분석에서 했던 것처럼, 한국어 감성분석에서는 새로 토크나이저 클래스를 상속을 받아서, 토크나이저를 재정의 해주어야 합니다.(그렇지 않으면 완전자모분리 현상 발생)
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
     for line in reader:
         token = line.strip()
         if "_" in token:
           token = token.replace("_","")
           token = "##" + token
         token_dict[token] = len(token_dict)
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

    return word

question = '파이썬을 만든 사람은 누구야?'
doc = '파이썬 로고 최근 몇 년 사이 프로그래밍을 비전공자들에게 알려주는 문화가 전 세계적으로 확산되고 있다. 하버드나 예일대 같은 해외 대학 뿐만 아니라 카이스트, 국민대, 성균관대 등 국내 대학에서도 프로그래밍 교양 수업이 늘어나는 추세다. 이러한 문화 속에서 함께 주목받는 언어가 있다. ‘파이썬’이다. 간결한 문법으로 입문자가 이해하기 쉽고, 다양한 분야에 활용할 수 있기 때문이다. 이 외에도 파이썬은 머신러닝, 그래픽, 웹 개발 등 여러 업계에서 선호하는 언어로 꾸준히 성장하고 있다.파이썬 로고 네덜란드 개발자가 만든 프로그래밍 언어파이썬은 네덜란드 개발자 귀도 반 로섬(Guido van Rossum)이 만든 언어다. 그는 암스테르담대학교에서 컴퓨터과학과 수학을 전공했으며, CWI(Centrum voor Wiskunde en Informatica, 국립 수학 및 컴퓨터 과학 연구기관)라는 연구소에 근무하면서 인터프리터 언어(interpreted language)를 개선하는 일을 맡게 됐다. 그러면서 CWI는 ‘ABC’라는 프로그래밍 언어를 팀원들과 새로 만들었다. ABC 언어 프로젝트가 시작된 지 4~5년이 지나자, CWI는 눈에 띄는 성과가 없다는 이유로 프로젝트를 종료시켰다. 이 과정에서 귀도 반 로섬은 같은 회사의 ‘아모에바’라는 팀으로 옮겨 마이크로 커널 기반 분산시스템 환경에 대해 연구하는 일을 진행했다.ABC 언어 개발과 분산시스템 연구는 파이썬을 만드는 데 중요한 영감을 주었다. 귀도 반 로섬은 “아모에바 프로젝트에 일하면서 하이레벨 언어가 필요하다는 것을 더욱 깨달았다”라며 “새로운 분산환경 시스템에 맞으면서 C와 셀에서 부족한 부분을 채워주는 새로운 언어를 개발하고 싶었다”라고 설명했다. 그렇게 해서 그는 취미활동으로 새로운 언어를 개발하기 시작했다.‘파이썬(Python)’이란 영어의 의미는 원래 그리스 신화에 나오는 뱀 이름이다. 파이썬 로고에 두 개의 뱀이 서로 마주본 듯한 그림이 있는 이유도 이 때문이다. 하지만 귀도 반 로섬이 실제 ‘파이썬’이란 단어를 선택할 당시에는 그리스 신화를 참고한 것은 아니었다. 영국 <BBC> 방송의 코미디 프로그램인 ‘몬티 파이썬 비행 서커스(Monty Python\'s Flying Circus)’를 좋아해 가져온 단어였다. 파이썬 공식 홈페이지에 따르면 “귀도 반 로섬은 짧고, 특별하고, 미스터리한 느낌의 단어를 찾았다”라며 “그래서 파이썬이란 이름을 결정했다”라고 설명돼 있다.파이썬 창시자 귀도 반 로섬 귀도 반 로섬은 1989년부터 본격적으로 파이썬을 개발하기 시작했고, 1990년 파이썬의 첫 버전을 공개했다. 처음 버전은 CWI 내 동료들이 대부분 이용했으며, 그들의 피드백을 거쳐 개선돼 왔다.1990년 이후에는 CWI가 아닌 외부에서 파이썬에 대한 소규모 세미나와 워크샵이 열리기 시작했다. 이때부터 몇몇 기업들은 파이썬을 실제 서비스에 하나둘 도입하기 시작했다. 귀도 반 로섬은 CWI 이후 CNRI(Corporation for National Research Initiatives), 비오픈닷컴, 잡코퍼레이션 등에 여러 단체와 기업에서 근무하며 파이썬만 전문적으로 개발했다. 이 과정에서 그는 파이썬에 대한 안정성을 높이고, 오픈소스 라이선스도 좀 더 유연하게 변경했다.현재 파이썬은 대형 글로벌 기업부터 스타트업까지 다양하게 안정적으로 활용되고 있다. 구글, 야후, 유럽 입자 물리 연구소(CERN), 미국항공우주국(NASA) 등이 파이썬을 이용해 서비스를 구축했다. 귀도 반 로섬은 2005년부터 아예 구글에 합류했으며, 약 7년 동안 구글에서 파이썬 관련 프로젝트를 이끌었다.실제로 구글은 파이썬을 많이 사용하는 기업으로 알려져 있다. 구글 내부에서 사용하는 코드리뷰 도구, ‘앱 엔진’ 같은 클라우드 제품 등이 파이썬을 이용해 만들어졌다. 귀도 반 로섬은 2012년 구글을 떠나 2013년부터 드롭박스(Dropbox)에 합류했다. 드롭박스에서는 현재 파이썬 언어를 개선하는 동시에 API 관련 개발을 진행하고 있다고 한다.파이썬 언어의 장·단점2016년 프로그래밍 인기 순위 1위로 파이썬이 꼽혔다. 코드 경진대회 서비스를 제공하는 코드이벨은 파이썬을 ‘2016년 프로그래밍 인기 순위 1위’로 꼽았으며, 프로그래밍 인기 순위를 매달 집계하는 레드몽크(RedMonk)나 티오베(Tiobe)도 파이썬의 인기 순위는 5위 안에 들어왔다. 그만큼 최근 인기가 높아지고 있다.파이썬은 어떤 장점이 있어 사람들에게 관심을 받을까? 먼저 파이썬은 문법이 간결하고 표현 구조가 인간의 사고 체계와 닮아 있다. 이 덕분에 초보자도 쉽게 배울 수 있다. 이러한 특징은 유지 보수와 관리도 쉽게 하도록 돕는다. 파이썬은 또한 외부에 풍부한 라이브러리가 있어 다양한 용도로 확장하기 좋다. 실제로 파이썬은 웹 개발 뿐만 아니라 데이터 분석, 머신러닝, 그래픽, 학술 연구 등 여러 분야에서 활용되고 있다. 생산성이 높은 것도 큰 장점이다.파이썬이 이용되는 분야들 파이썬 디자인 원리라고 불리는 ‘젠 오브 파이썬(Zen of Python)’을 보면 파이썬이 추구하는 가치를 보다 자세히 알 수 있다.파이썬 선(禪)(Zen of Python)아름다움이 추함보다 좋다.(Beautiful is better than ugly.)명시가 암시보다 좋다.(Explicit is better than implicit.)단순함이 복잡함보다 좋다.(Simple is better than complex.)복잡함이 꼬인 것보다 좋다.(Complex is better than complicated.)수평이 계층보다 좋다.(Flat is better than nested.)여유로운 것이 밀집한 것보다 좋다.(Sparse is better than dense.)가독성은 중요하다.(Readability counts.)특별한 경우라는 것은 규칙을 어겨야 할 정도로 특별한 것이 아니다.(Special cases aren\'t special enough to break the rules.)허나 실용성은 순수성을 이긴다.(Although practicality beats purity.)오류는 절대 조용히 지나가지 않는다(Errors should never pass silently.)명시적으로 오류를 감추려는 의도가 아니라면.(Unless explicitly silenced.)모호함을 앞에 두고, 이를 유추하겠다는 유혹을 버려라.(In the face of ambiguity, refuse the temptation to guess.)어떤 일에든 명확한 - 바람직하며 유일한 - 방법이 존재한다.(There should be one? and preferably only one ?obvious way to do it.)비록 그대가 우둔하여 그 방법이 처음에는 명확해 보이지 않을지라도.(Although that way may not be obvious at first unless you\'re Dutch.)지금 하는게 아예 안하는 것보다 낫다.(Now is better than never.)아예 안하는 것이 지금 당장보다 나을 때도 있지만.(Although never is often better than right now.)구현 결과를 설명하기 어렵다면, 그 아이디어는 나쁘다.(If the implementation is hard to explain, it\'s a bad idea.)구현 결과를 설명하기 쉽다면, 그 아이디어는 좋은 아이디어일 수 있다.(If the implementation is easy to explain, it may be a good idea.)네임스페이스는 대박 좋은 아이디어다 -- 더 적극적으로 이용해라!(Namespaces are one honking great idea?let\'s do more of those!)물론 파이썬도 단점이 있다. 예를 들어 속도가 느리다는 평가도 있으며, 모바일 앱 개발 환경에서 사용하기 힘들다. 또한 컴파일 시 타입 검사가 이뤄지지 않아 개발자가 실수할 여지가 조금 더 많다거나 멀티코어를 활용하기 쉽지 않다는 지적도 있다.파이썬 재단과 커뮤니티한국에서 2016년 8월 개최된 파이콘 APAC 행사(세계 각국의 파이썬 커뮤니티에서 주관하는 비영리 컨퍼런스) 기념품. 파이썬 커뮤니티는 참여자들에게 개방적이고 친절한 태도를 가진 것으로 유명하다. 파이썬 소프트웨어 재단(Python Software Foundation, PSF)은 2001년에 설립됐다. 파이썬 재단은 비영리 단체다. 주로 라이선스, 법적 분쟁, 도메인 주소, 후원금, 컨퍼런스 지원, 교육 활동 등을 관리하고 있다. 행정 업무를 관리하는 직원부터 이사회 멤버, 펠로우 등 다양한 사람들이 파이썬 재단에 참여하고 있다. 파이썬 재단 이사회 멤버는 현재 11명인데, 2016년 6월 한국인 개발자 김영근 이사가 최초로 이사회 멤버로 선출되기도 했다. 이사회는 자원봉사 형태로 무보수로 일하며, 행정적인 부분부터 재단의 방향과 관련된 다양한 문제를 회의를 통해 결정한다.파이썬 커뮤니티는 다양한 사람들에게 열린 태도를 가진 커뮤니티로도 유명하다. 사실 일부 오픈소스 커뮤니티는 누구나 기술 기여가 가능함에도 불구하고 새로운 사람이 합류되기 어렵다거나 기존 기득권 세력에 맞춰 분위기가 좌지우지되기도 한다. 파이썬 커뮤니티는 아예 재단 차원에서 ‘다양성 성명서’를 발표해 나이, 성별, 종교, 인종 등과 관련된 차별적인 발언을 지양하고 있다. 또한 입문자를 위한 세미나도 자주 개최하면서 새로운 사람 및 다양한 사람들이 함께 할 수 있는 문화를 만드는 데 많은 노력을 기울이고 있다.파이썬 오픈소스 기술 자체는 ‘자비로운 종신독재자(BDFL, Benevolent Dictator for Life)’ 체제로 운영되고 있다. BDFL은 오픈소스 커뮤니티에서 사용되는 용어다. 소스코드를 수정하는 최종 권한을 갖거나 전체적인 개발 방향을 정해주는 사람을 말한다. 주로 해당 오픈소스 기술을 처음 만든 창시자가 BDFL을 맡는다. 따라서 파이썬 커뮤니티에서도 수많은 개발자가 소스코드 개선에 참여하지만 최종적인 수정 권한은 귀도 반 로섬이 결정한다.'
word = predict_letter(question, doc)
print(type(word))
