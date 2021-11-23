## MS COCO 데이터를 한글캡셔닝으로 학습시킨 모델 + 멀티모달 데이터 추가활용
## 멀티모달 한글 캡셔닝 전처리 : hanspell(맞춤법) 수정
### 텐서플로우 케라스 토크나이저를 사용하였으며, 세부내용은 아래와 같습니다.

# 학습 데이터 수 : 147261장(사진), 736582문장(캡션수)
# 데이터셋 분할 : 훈련 : 검증 : 시험 / 8 : 1 : 1
# vocab_size = 27541 / max_length = 58
# Batch_size = 64, Buffer_size = 1000, embedding_dim = 256
# units = 512 / features_shape = 2048 / attention_features_shape = 64

# Epoch 1 : Loss 0.543391 / Loss(val) 1.484025
# Epoch 2 : Loss 0.481019 / Loss(val) 1.494184
# Epoch 3 : Loss 0.459880 / Loss(val) 1.495114
# Epoch 4 : Loss 0.445534 / Loss(val) 1.503034
# Epoch 5 : Loss 0.434286 / Loss(val) 1.526546
# Epoch 6 : Loss 0.425108 / Loss(val) 1.540671
# Epoch 7 : Loss 0.417466 / Loss(val) 1.539157
# Epoch 8 : Loss 0.410838 / Loss(val) 1.556598
# Epoch 9 : Loss 0.405163 / Loss(val) 1.543960
# Epoch 10 : Loss 0.400042 / Loss(val) 1.575294
# Epoch 11 : Loss 0.396082 / Loss(val) 1.556792
# Epoch 12 : Loss 0.391692 / Loss(val) 1.561010
# Epoch 13 : Loss 0.388281 / Loss(val) 1.557776
# Epoch 14 : Loss 0.384969 / Loss(val) 1.557019
# Epoch 15 : Loss 0.382431 / Loss(val) 1.578591
# Epoch 16 : Loss 0.379649 / Loss(val) 1.583895
# Epoch 17 : Loss 0.377440 / Loss(val) 1.582531
# Epoch 18 : Loss 0.375053 / Loss(val) 1.576649
# Epoch 19 : Loss 0.372932 / Loss(val) 1.581084
# Epoch 20 : Loss 0.371102 / Loss(val) 1.583513
# Epoch 21 : Loss 0.369532 / Loss(val) 1.594376
# Epoch 22 : Loss 0.367874 / Loss(val) 1.602497
# Epoch 23 : Loss 0.367510 / Loss(val) 1.584829
# Epoch 24 : Loss 0.365204 / Loss(val) 1.606700
# Epoch 25 : Loss 0.364498 / Loss(val) 1.600331

# ------------------------ 모델 구현 파트 ------------------------ 

# 필요 패키지 임포트
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
from konlpy.tag import Okt

# 모델 학습시 사용한 변수 값들 로드
# 해당 모델에 들어가야하는 필수 변수들이며, 학습시킬때의 수치와 일치화 시켜주었습니다.
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = 27541
max_length = 58
features_shape = 2048
attention_features_shape = 64

# 토큰나이저 파일 로드 및 설정
PATH = 'C:/Workspace/python/빅데이터 지능형서비스 개발 팀프로젝트/Final Project/Data/IC_Data/Model_Data/Captioning/Tensor_Ko_TK_Okt/'
df_Tk = pd.read_csv(PATH + 'tokenizer(word_to_index_Okt).csv')
tokenizer = Okt()
word_to_index = dict(zip(df_Tk['word'], df_Tk['index']))
index_to_word = dict(zip(df_Tk['index'], df_Tk['word']))

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

    # dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    dec_input = tf.expand_dims([word_to_index['start']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(index_to_word[predicted_id])

        # if tokenizer.index_word[predicted_id] == '<end>':
        #     return result, attention_plot
        if index_to_word[predicted_id] == 'end':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
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