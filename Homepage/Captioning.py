# ------------------------ 모델 설명 ------------------------ 
## 코드는 구글 텐서플로우 튜토리얼 코드를 일부변형하여 작성하였습니다.
## https://www.tensorflow.org/tutorials/text/image_captioning?hl=ko
## MS COCO 데이터를 한글캡셔닝으로 학습시킨 모델 + 멀티모달 데이터 추가활용
## 멀티모달 한글 캡셔닝 전처리 : hanspell(맞춤법) 수정
### 텐서플로우 케라스 토크나이저를 사용하였으며, 세부내용은 아래와 같습니다.

# 학습 데이터 수 : 147261장(사진), 736582문장(캡션수)
# 데이터셋 분할 : 훈련 : 검증 : 시험 / 8 : 1 : 1
# vocab_size = 76344 / max_length = 41
# Batch_size = 64, Buffer_size = 1000, embedding_dim = 256
# units = 512 / features_shape = 2048 / attention_features_shape = 64

# Epoch 1 : Loss 0.834422 / Loss(val) 1.313634
# Epoch 2 : Loss 0.489833 / Loss(val) 1.398128
# Epoch 3 : Loss 0.595815 / Loss(val) 1.683201
# Epoch 4 : Loss 0.563709 / Loss(val) 1.687328
# Epoch 5 : Loss 0.537522 / Loss(val) 1.700229
# Epoch 6 : Loss 0.516654 / Loss(val) 1.706951
# Epoch 7 : Loss 0.499847 / Loss(val) 1.684718
# Epoch 8 : Loss 0.485388 / Loss(val) 1.732121
# Epoch 9 : Loss 0.474416 / Loss(val) 1.731298
# Epoch 10 : Loss 0.463986 / Loss(val) 1.735539
# Epoch 11 : Loss 0.457107 / Loss(val) 1.735508
# Epoch 12 : Loss 0.447212 / Loss(val) 1.737649
# Epoch 12 : Loss 0.447212 / Loss(val) 1.737649
# Epoch 14 : Loss 0.433879 / Loss(val) 1.744081
# Epoch 15 : Loss 0.428988 / Loss(val) 1.747652
# Epoch 16 : Loss 0.423603 / Loss(val) 1.755263
# Epoch 17 : Loss 0.419722 / Loss(val) 1.757329
# Epoch 18 : Loss 0.414604 / Loss(val) 1.762197
# Epoch 19 : Loss 0.409950 / Loss(val) 1.773284

# ------------------------ 모델 구현 파트 ------------------------ 

# 필요 패키지 임포트
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
import pandas as pd
from hanspell import spell_checker

# 모델 학습시 사용한 변수 값들 로드
# 해당 모델에 들어가야하는 필수 변수들이며, 학습시킬때의 수치와 일치화 시켜주었습니다.
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = 76344
max_length = 41
features_shape = 2048
attention_features_shape = 64
top_k = vocab_size - 1

# 토큰나이저 파일 로드 및 설정
PATH = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Data/IC_Data/Model_Data/Captioning/Tensor_Ko_TK_Keras/'
df_Tk = pd.read_csv(PATH + 'tokenizer(index_word).csv')
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
tokenizer.word_index = dict(zip(df_Tk['value'], df_Tk['key']))
tokenizer.index_word = dict(zip(df_Tk['key'], df_Tk['value']))

# 모델구동시 필요한 함수 로드
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# 모델 작성
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 모델에 학습시킨 가중치 적용
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, PATH, max_to_keep=50) 
ckpt.restore(ckpt_manager.latest_checkpoint)

# 출력시킬 함수 작성
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    result = spell_checker.check(result)
    result = result.checked
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result/2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()