---
layout: post_without_img
title: "[WIP]2019 TF dev summit (1) - Keynote"
author: "Sangsu Lee"
categories: journal
tags: [documentation,sample]
image: tf-logo.png
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/b5Rs1ToD9aI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 2019 TF dev summit series
Tensorflow Dev Summit 2019 발표 영상들을 코드와 함께 요약하고 있습니다.
오역이나 의역, 빠진 내용들이 있을 수 있으며 더 자세한 컨퍼런스 정보는 [Tensorflow Dev Summit Event page](https://www.tensorflow.org/dev-summit) 에서 확인하실 수
있습니다.  

**[(1) - Keynote]()**  
[02. Airbnb uses machine learning to help categorize its listing photos]()


<br />

## Keynote - Megan Kacholia
Machine learning은 지금도 전례없는 혁신을 경험하고 있고, 
이를 크게 컴퓨팅 자원, 알고리즘, 많은 데이터 세 가지 요인들의
결과로 생각할 수 있습니다. Tensorflow를 사용해 실제 문제를 해결한 좋은 사례들을 몇 가지 소개하겠습니다.

- 인도 델리 지역에서의 대기오염 문제가 심각한데, 센서를 설치하는데 드는 비용이 큼.  
 => 델리의 학생들이 스마트폰에서 하늘 사진을 찍으면 대기 오염 정도를 근사해주는,
**Aircognizer**라는 스마트폰 어플리케이션을 개발함.  

- 많은 사용자들에게 개인화된 최적의 컨텐츠만을 제공하기 위해 Twitter는 
**Ranked timeline**을 활용, Tensorflow Hub, Tensorboard와 같은 툴들로
기업의 한정된 자원을 효율적으로 사용하고 모델의 성능도 높일 수 있게 됨.

- G-healthcare사는 의료 영상(MRI) 촬영시 Tensorflow model을 
실시간으로 접목해 **촬영에서 환자가 누워있는 각도와 방향을 측정하는 기술로 활용**해 촬영 시간을 줄였고
측정 오류도 감소시킬 수 있게 됨.  

<br />

## Tensorflow 2.0 - Rajat Monga

감사하게도 Tensorflow의 여러 사용자들이 라이브러리의 어떤 부분이 좋은지
혹은 어떤 부분이 싫은지와 같은 피드백을 계속 내 주었습니다.

- 사용자들은 더 쉬운 API, 더 직관적인 API를 요구했고, 
- 라이브러리의 높은 Complexity와 Redundancy를 지적했으며,
- 공식 문서와 예제 코드들의 Improvement를 요구했습니다.
그리고 이것들이 Tensorflow 2.0을 만들며 저희가 가장 중점적으로 생각해왔던 것이기도 합니다.  

### Easy, Powerful, Scalable
![image](https://user-images.githubusercontent.com/25409073/54100867-fd7e1500-4403-11e9-9658-73c3e7c5fc97.png)  

라이브러리를 더 사용하기 쉽게 만들기 위해 **Keras**에 주목했고, 파이썬의 단순성과 어울리는 **Eager execution**을 도입했습니다.
말도 안되는 아이디어를 코드로 구현할 수 있는 유연함과, exaFLOPS(초당 10^18번의 연산)을 뛰어넘을 수 있는 
능력을 갖게 됨으로써 Tensorflow는 그 어느 때보다 더 강력해졌습니다.
또 개발에서의 결과물들을 Scalable하게 배포할수 있으며 구글 스케일의 테스트 환경이 이 구조를 견고하게 지탱하고 있습니다.
이제 Tensorflow의 전체적인 아키텍쳐를 살펴보겠습니다.  
<br />

### Components in Tensorflow
![image](https://user-images.githubusercontent.com/25409073/54100877-0f5fb800-4404-11e9-9246-04f74be46173.png)  

훈련부터 모델 배포까지 필요한 많은 컴포넌트들과 기능들이 계속해서 추가되어 왔습니다.
Tensorflow 2.0부터는, 이전보다 훨씬 더 유기적이고 효율적으로 컴포넌트들이 구성되고 실행된다는 것을 강조드리고 싶습니다.  
<br />

### Training Workflow
![image](https://user-images.githubusercontent.com/25409073/54100888-1be41080-4404-11e9-8f96-304a73ee0fe3.png)  

이 그림은 다양한 컴포넌트들이 전체 훈련 과정 속에서 어떻게 결합되는지 보여줍니다.
`tf.data`는 데이터셋의 입력 파이프라인과 전처리를, **Keras**와 **Premade estimator**가 
모델 구축에 관여합니다. 그 다음 **Eager Execution**과 **Graph**로 훈련시키고, 
`SavedModel`로 배포를 위해 모델을 패키징합니다. 다음은 단순한 Traning workflow의 예시입니다:  
```python
# Load data
import tensorflow_dataset as tfds

dataset = tfds.load('fashion_mnist', as_supervised=True)
mnist_train, mnist_test = dataset['train'], dataset['test']

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

mnist_train = mnist_train.map(scale).batch(64)
mnist_test = mnist_test.map(scale).batch(64)
```

새로운 아이디어의 검증을 위해 여러 public 데이터셋을 실험해 보고 싶을 때가 많습니다.
이미 잘 알려진 데이터셋은 물론이고 빠르게 성장하고 있는 
데이터셋 또한 `tensorflow_datasets`로 아주 쉽게 불러와 바로 모델링을 시작할 수 있도록 해 줍니다.
`tf.data`로 임의의 데이터셋을 직접 구축하는 것 또한 훨씬 단순해집니다.
<br />

### Training Workflow - tf.keras
그리고 마치 우리가 모델을 생각하는 방식처럼, Keras를 사용해 모델을 Layer로 표현할 수 있습니다. 
```python
import tensorflow as tf
mnist_train = ...
mnist_test = ...

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(mnist_train, epochs=5)
model.evaluate(mnist_test)
```
### Training Workflow - MirroredStrategy
보통 딥러닝 모델은 아주 많은 연산량이 필요합니다.
한 대의 머신에서 실행하는 것보다, `MirroredStrategy` 모듈과 스코프를 단 몇 줄의 코드만 추가하면
여러 대의 머신에 연산을 분산시켜 처리하는 전략을 쉽게 취할 수 있습니다.
위의 코드로부터는 두 줄이 추가됩니다:
```python
import tensorflow as tf
dataset = ...

# strategy object 초기화, 스코프 정의
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    model = tf.keras.models.Sequential([
        ...
    ])
    model.compile(...)

model.fit(...)
model.evaluate(...)
```

### Pretrained network with Tensorflow Hub
(gif그림)
Pretrained model로부터 시작하는 것 또한 컴퓨팅 자원을 아끼는데 도움이 됩니다.
Tensorflow Hub는 설계하고자 하는 모델에 방대한 종류의 Pretrained component를 끼워서 사용할 수 있고,
또 특정한 데이터셋에 맞게 fine-tune 시킬 수도 있습니다.  
<br />

### Subclassing, Custom Loop
![image](https://user-images.githubusercontent.com/25409073/54120315-c119db80-443a-11e9-98bf-c431ba682b52.png)  

`tf.keras`와 `tf.estimator`는 High-level API로써의 Building block을 제공합니다.
더불어 딥러닝 모델의 훈련 과정에 일반적으로 많이 쓰이는 모듈들도 함께 제공되지만,
가끔은 저수준의 컨트롤 또한 필요할 때가 있습니다. 이 때 **sub-classing**, 그리고 **custom loop**를 통해 
조금 더 세부적인 부분까지 수정하며 구축할 수 있습니다. 
다음은 Subclassing의 예제입니다. 기계 번역에서의 Custom Encoder를 
만들고 싶다면:
```python
# Sub-classing example
# inherit tf.keras.Model class as a parent
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,
                 enc_units, batch_sz):

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
```
이렇게 사용자가 서브클래스에서 꼭 구현해야 하는 함수들(`__init__()`, `call()`)만 정의하면 
외적인 부분은 프레임워크에서 맡아 처리하게 되므로 Computation algorithm을 구현하는데 집중할 수 있게 됩니다. 


또 Training loop를 커스터마이징해 Gradient와 Optimization 과정 또한 원하는 대로 바꿀 수 있습니다.
```python
# Custom loops example
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']]*BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            pred, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], pred)

            dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
```

### Tensorboard
(tensorboard gif) 
Keras로 작성한 모델이나, 더 복잡한 모델들도 훈련 과정, 결과를 보고 분석하는 것이
굉장히 중요할 수 있습니다. 이를 위해 Tensorboard가 다양한 시각화를 제공하며
Colab과 Jupyter notebook에서 Tensorboard의 통합된 인터페이스를 볼 수 있게 됩니다.


### Tensorflow 2.0
![image](https://user-images.githubusercontent.com/25409073/54137148-701cde00-4460-11e9-83f8-c7dcc22579f4.png)  

앞서 말씀드린 모든 기능들을 Tensorflow 2.0에서 만나볼 수 있으며
Alpha version을 설치해 지금 바로 사용할 수 있습니다. 
**또한 Tensorflow 1.X 버전의 코드를 업그레이드하기 위한 변환 스크립트가 제공되고
1.X 버전의 API들도 여전히 사용가능합니다.** 그리고 저희는 2019년 2분기에 완전한 Tensorflow 2.0의 
정식 릴리즈를 목표로 하고 있습니다.  
<br />

## Research, Prototyping, Production - Megan Kacholia
![image](https://user-images.githubusercontent.com/25409073/54139897-dfe19780-4465-11e9-9eac-8420214def5c.png)  

연구자들은 많은 State-of-the-art 논문의 실험에 Tensorflow를 사용하고 있으며,
**Eager execution**은 여러 실험들에 꼭 필요한 유연성을 제공하게 됩니다.
Tensorflow 2.0에서는 모든 Python 커맨드가 즉시 실행되며, Define-and-run 방식이 아닌 
우리가 익숙한 방식으로 코드를 작성하고 바로 실행할 수 있게 됩니다.
또 코드 작성 뿐만 아니라 디버깅도 굉장히 쉬워집니다.
```python
def f(x):
    while tf.reduce_sum(x) > 1:  # Control flow runs eagerly
        x = tf.tanh(x)
    return x

f(tf.random.uniform([10])  # Immediately outputs a value
```

Eager mode에서 개발을 마치고 나면 최종적으로는 모델을 GPU, TPU, 혹은 다른 하드웨어에
올려 배포하게 됩니다. `@tf.function`은 Eager mode에서 Graph로의 변환을 도와주는 모듈입니다.
개발할 때는 Eager mode로 python control flow, `print`구문, `assert`구문 등 어떤 직관적인 툴을 사용하다가도
언제든지 필요할 때 함수를 Graph로 변환할 수 있게 됩니다. 다음은 `@tf.function` 데코레이터의 사용 예입니다:
```python
@tf.function
def f(x):  # And now this function will become a graph
    ...
```




