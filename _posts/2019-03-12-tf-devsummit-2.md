---
layout: post_without_img
title: "2019 TF dev summit(2) - Introducing Tensorflow 2.0 and its high-level APIs"
author: "Sangsu Lee"
categories: journal
tags: [documentation,sample]
image: tf-logo.png
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/k5c-vg4rjBw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  

### 2019 TF dev summit series
Tensorflow Dev Summit 2019 발표 영상들을 코드와 함께 요약하고 있습니다.
오역이나 의역, 빠진 내용들이 있을 수 있으며 더 자세한 컨퍼런스 정보는 [Tensorflow Dev Summit Event page](https://www.tensorflow.org/dev-summit) 에서 확인하실 수
있습니다.  

[01 - Keynote]({% post_url 2019-03-10-tf-devsummit-1 %})  
**[02 - Introducing Tensorflow 2.0 and its high-level APIs]()**  
03 - Airbnb uses machine learning to help categorize its listing photos  
04 - tf.function and Autograph  
05 - Tensorflow Datasets  
06 - What's new in Tensorboard  
07 - Tensorflow.js 1.0  
08 - Utilizing Deep Learning to better predict extreme weather  
09 - Tensorflow Hub: Reusable Machine Learning  
10 - Tensorflow Probability: Learning with confidence  
11 - Tensorflow Extended(TFX) Overview and Pre-training Workflow  
12 - Tensorflow Extended(TFX) Post-training Workflow  
13 - Building a Visual Debugging Tool for ML - TF.js in Interactive Visual Analytics  
14 - Improving Text in Tensorflow  
15 - Upgrade your existing code for Tensorflow 2.0  
<br />

## Introducing Tensorflow 2.0 - Martin Wicke
![image](https://user-images.githubusercontent.com/25409073/54407136-029ed500-4720-11e9-9a83-44e836f0e3fd.png)  

### Painful moments
Tensorflow는 가끔씩 사용하기 조금 고통스러웠고 어려웠습니다.
![image](https://user-images.githubusercontent.com/25409073/54179727-fecf4080-44dc-11e9-9e1a-0989133cde8b.png)  

위처럼 **Session**을 사용하는 것은 평범한 파이썬 유저인 우리들에게 너무 부자연스러웠고
Tensorflow가 성장하면서도 계속해서 라이브러리가 복잡해지고 헷깔리기 시작했습니다. 
Tensorflow로 아주 많은 것을 할 수 있었지만, Tensorflow를 **어떻게** 사용하는 것이
가장 잘 사용하는 것인지는 확실하지 않았습니다.

![image](https://user-images.githubusercontent.com/25409073/54178470-448a0a00-44d9-11e9-92ce-b942ff8d0141.png)  

Tensorflow가 발전하면서 저희도 정말 많이 배웠습니다. 사용자 입장에서의 빠른 프로토타이핑, 
그리고 쉬운 디버깅이 절실하게 필요하다는 것을 느꼈습니다. 

### Install
![image](https://user-images.githubusercontent.com/25409073/54215568-7fb52900-452b-11e9-8575-5dad799f41fd.png)  

많은 것들이 보완된 Tensorflow 2.0 Alpha version가 오늘 릴리즈됐고, 이 커맨드로 바로 설치해 실행할 수 있습니다.
그래서, 어떤 점이 바뀌었을까요?  
<br />

### Usability, Clarity, Flexibility
![image](https://user-images.githubusercontent.com/25409073/54215618-93608f80-452b-11e9-99ee-9482d0d5d241.png)  

저희가 Tensorflow 2.0을 만들 때 가장 중점적으로 고려했던 부분이 바로 사용의 편리함, Usability입니다.
1. **Keras**를 Tensorflow의 High-level API로 가져왔고, 또 Tensorflow와 깊이있게 병합했습니다.
더 나아가 Tensorflow의 Advanced feature들까지 `tf.keras` 안에서 사용할 수 있도록 했습니다.  
2. 또다른 중요한 변화는 **Eager Execution**의 디폴트 실행입니다.
Eager execution 이전의 Tensorflow는 **Declarative style**로 Computation graph를 정의한 뒤 실행되지만,
Tensorflow 2.0는 한 줄씩 실행되는 익숙한 파이썬 코드 스타일을 생각하면 됩니다.  
<br />

그래서, 이렇게 두 숫자를 더하게 되면:
```bash
>>> tf.add(2, 3)
<tf.Tensor: id=2, shape=(), dtype=int32, numpy=5>
```
결과를 바로 확인할 수 있습니다. 세션도, 그래프도 생각할 필요가 없습니다.
Tensorflow가 그래프를 버린걸까요? 사실 그렇지는 않습니다.
**Serialization**, **쉬운 배포**, **분산 처리**와 **Optimization** 등 Graph의 장점을 버릴 수는 없습니다. 
단지 조금 더 쉬워졌을 뿐입니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/54215656-a2474200-452b-11e9-9dbe-7ae31ecef6cc.png)  

Tensorflow 2.0은 Major release이므로, 이제 버릴 것들은 버려야 했습니다. 중복된 기능들을 삭제했고 API들을 통합했습니다.
API가 사용자들에게 더 일관성있게 보일 수 있도록 신경썼으며, Tensorflow 뿐만 아니라 Tensorflow와 함께 성장하고 있던 
모든 Ecosystem들에게도 이런 변화들이 적용되었습니다. 모든 Ecosystem에서 추가적인 변환작업이 없이 통용될 수 있는 
모델의 **Exchange format**도 정의했습니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/54215690-b3904e80-452b-11e9-9906-2e506f14f454.png)  

그렇게 2.0 릴리즈에서 많은 것들을 버리게 되면서 Tensorflow는 훨씬 더 유연해집니다.
low-level API, `tf.raw_ops`가 Tensorflow의 모든 내부 연산들을 조작할 수 있게 해 줄 것이고,
`variables`, `checkpoints`, `layers`, 이외의 많은 Tensorflow 주요 컨셉들의 클래스를 상속받는 방법으로
각 컨셉의 Custom class를 만드는 것이 가능합니다.

### Upgrade to TF 2.0
![image](https://user-images.githubusercontent.com/25409073/54191307-fdac0c80-44f8-11e9-9662-9f2f9a9cd366.png)
그렇다면 이제 우리의 질문은 _'어떻게 업그레이드를 하지?'_ 가 됩니다.
어떤 버전 업그레이드라도 그 과정이 쉽지 않음을 우린 잘 알고 있고, 특히 이런 Major upgrade의 경우 더욱 그렇습니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/54215799-e0dcfc80-452b-11e9-9afb-a3ecd6ccc881.png)
- 사실, 최근 구글 또한 엄청난 규모의 코드베이스에 대해 TF 2.0으로의 업그레이드를 시작했습니다.
이 업그레이드가 진행되는 동안 많은 양의 Migration guide, 그리고 Best practice들이 공개될 것이고 
우리가 구글에서 하고 있는 것처럼 여러분들 또한 충분히 할 수 있을 것이라고 생각합니다.
- 또 2.0으로의 마이그레이션을 위해 굉장히 많은 툴들을 제공합니다. 가장 먼저
Tensorflow 2.0의 일부분으로써 `tf.compat.v1` 모듈이 탑재됩니다. 2.0에서 사용할 수 없게 된 함수를
꼭 사용해야 하는 경우라면 이 패키지에서 이전에 사용했던 함수를 찾을 수 있습니다.
**하지만, `tf.contrib` 패키지는 포함되지 않습니다.**
- 그리고, Tensorflow 1.x으로 작성한 스크립트 파일을 바로 변환시켜주는 `tf_upgrade_v2` 커맨드 라인 툴을 사용하면
2.0에서 바뀐 함수명으로의 변경이나 함수 Argument 순서의 변경 등 버전 변경사항을 자동으로 스크립트에 적용해 줍니다.
이 때, 2.0부터 삭제되거나 Deprecated된 함수를 사용하고 있었다면, 위에서 설명한 `tf.compat.v1`패키지의 함수로 변환해 주고,
`tf.contrib` 모듈을 사용했던 경우라면 변환이 실패할 수 있습니다.  
<br />

### tf_upgrade_v2
![image](https://user-images.githubusercontent.com/25409073/54195977-71531700-4503-11e9-99db-9fe8e4383778.png)  


```bash
# Example file conversion in command line
$ tf_upgrade_v2 --infile cnn_model.py --outfile cnn_model_upgraded.py
```
위와 같이 변환 결과를 커맨드 라인에서 볼 수 있습니다.
주의할 점은 이 자동변환 툴 `tf_upgrade_v2`는 1.x버전의 코드를 2.0에서도 문제없이 실행될 수 있도록 고쳐주지만,
코딩 스타일을 Tensorflow 2.0의 스타일로 고쳐주지는 않는다는 점입니다.  
<br />

## High level APIs in Tensorflow 2.0 - Karmel Allison
![image](https://user-images.githubusercontent.com/25409073/54407610-d421f980-4721-11e9-9242-d7f7973d4615.png)  

### High-level API

High-level API의 개념에 대해 한번 생각해 볼 필요가 있습니다.
기둥을 세우고, 벽을 세우고, 지붕을 덮듯이 우리가 집을 지을 때처럼
Machine learning 모델을 만들때도 공통적으로 거치는 과정들이 있습니다. 
이와 유사한 컨셉으로 High-level API는 모델을 더 쉽고, 더 빠르게 만들어 확장할 수 있도록 도와줍니다.

### tf.keras
![image](https://user-images.githubusercontent.com/25409073/54201438-00b2f700-4511-11e9-920e-36d630eeff21.png)
2년 전, Keras는 Tensorflow에서 사용할 수 있는 High-level API들 중 하나로 채택되었습니다.
Keras는 본질적으로 **Model building**을 위한 Specification인데, 다른 머신러닝 프레임워크들과도
잘 호환되며 모델의 Layer, model, loss, optimizer들을 정의하는 공통의 언어로 자리매김했습니다.
저희는 Tensorflow에 최적화된 **Tensorflow version의 Keras**를 만들었고, 이를 `tf.keras`라고 이름지었습니다.

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
Keras는 그 자체로 Pythonic하고 또 배우기 쉽도록 만들어져 사람들을 
Machine learning의 세계로 더 쉽게 끌어당길 수 있습니다.
위 코드는 전체 모델의 완전한 구조와 Training loop를 보여줍니다. 
이렇게 사용자들이 장황한 코드를 작성할 필요 없이 쉽게 모델의 구조를 
바꾸거나 다시 만들 수 있습니다.

```python
# Custom keras model example
class MyModel(tf.keras.Model):
  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)
```
이와 더불어 상속과 인터페이스를 활용해 Keras를 굉장히 유연하게 사용할 수 있습니다.
위와 같이 `tf.keras.Model`을 상속받은 Subclass에서 임의의 모델 구조를 정의할 수 있고,
원한다면 각 Training step마다의 진행도 커스터마이징이 가능합니다. 
Keras는 간단하고 효율적이며 누구라도 API를 어떻게 사용하는지 잘 이해할 수 있습니다.  

하지만 한 가지 문제가 있습니다. **tf.keras는 단순한 대신에, 비교적 크기가 작은 
모델을 가정하고 만들어졌기 때문에 거대한 Google 내부 시스템과 같은 딥러닝 
모델 환경에서는 적합하지 않을 수 있었습니다.**

### tf.estimator
```python
wide_columns = [
  tf.feature_column.categorical_column_with_identity(
    'user_id', num_buckets=10000)]

deep_columns = [
  tf.feature_column.numeric_column('visits'),
  tf.feature_column.numeric_column('clicks')]

tf.estimator.DNNLinearCombinedClassifier(
  linear_feature_columns=wide_columns,
  dnn_feature_columns=deep_columns,
  dnn_hidden_units=[100, 75, 50, 25])
```
네트워크의 크기가 큰 모델에서는 **Estimator**가 더 나은 선택이 될 수 있습니다.
Estimators는 아주 많은 머신들 위에서의 Scalable하고 더 견고한 실행을 가능하게 합니다.
위 코드는 tf.keras를 사용한 예시보다 더 깊고 넓은 구조를 가지며, 
이는 Estimator가 더 효율적으로 활약할 수 있는 환경입니다.  

Estimator는 아주 강력한 기계에 비유할 수 있습니다. 하지만 **자유도가 높지 않으며** 
**문법에 익숙해지는것이 쉽지 않다**는 의견이 많았습니다. 또 Estimator의 어떤 부분이 어느 부분과 
연결되는지 알기 힘든 단점도 있었습니다.

### tf.keras & tf.estimator
![image](https://user-images.githubusercontent.com/25409073/54211749-2f3acd00-4525-11e9-9e47-4fb908e0fefc.png)  

지난 2년동안 `tf.keras`와 `tf.estimator`를 함께 개발하며 많은 것들을 배웠습니다.
그리고 우리는, 사용자에게 **Simple API와 Scalable API 둘 중 하나를 선택하도록 하는 것이
바람직하지 않다**는 결론을 내리게 되었습니다.
우리는 간단한 MNIST부터 천문학적 스케일까지 포용할 수 있는
**단 하나의 High-level API**를 원했습니다.

### tf.keras, 2.0
![image](https://user-images.githubusercontent.com/25409073/54213258-788c1c00-4527-11e9-9bcb-4f09e8b26621.png)  

그래서 Tensorflow 2.0에서는, **Keras API로 모델 구축을 표준화합니다.**
그러면서도 tf.estimator의 확장성과 견고함을 가져와 Prototyping - Distributed Training - 
Production Serving에 이르기까지의 모든 과정을 한번에 해결하도록 했습니다.

### tf.keras in TF 1.x, TF 2.x
다음은 Tensorflow 1.x와 2.0에서의 tf.keras 문법입니다.
![image](https://user-images.githubusercontent.com/25409073/54213992-c3f2fa00-4528-11e9-875d-c2b9feb71eea.png)  
알아채신 분도 계시겠지만, 두 코드는 완벽히 동일합니다.
그렇다면, tf.keras의 문법은 1.x에서 넘어오면서 어떤 점이 바뀌었을까요?

사실 Tensorflow 2.0에서 새롭게 제공하는 모든 기능들을 tf.keras의 인터페이스 안에서 합치기 위해
정말 많은 노력들이 있었습니다. 예를 들어 위의 두 코드는 완벽히 일치하더라도, 1.13버전에서는 Graph 기반의 모델이
세션 안에서 생성되고 실행되는 반면, 2.0에서는 같은 모델이 Eager mode에서 실행됩니다.

### Debugging with Eager
그렇게 되면서, Eager mode의 장점을 온전히 tf.keras의 context 안에서 사용할 수 있게 됩니다.
```python
data = tf.data.TFRecordDataset(['file1', 'file2'])
data = data.map(parse_fn).batch(32)

for row in data.take(3):
  print(row)

>>> (<tf.Tensor: id=38, shape=(32, 32, 28), dtype=float64, numpy=
    array([[ 0.070588,  0.494117,  0.533333,  0.686274,
             0.650980,  1.      ,  0.968627,  0.498039]])>...
model.fit(data, epochs=1)
```
Input pipeline에서의 dataset이 보통의 **numpy array**처럼 행동하는 것을
볼 수 있습니다. 디버깅이 쉬워지며 Keras model에도 아주 자연스럽게 끼어들어가게 됩니다.

### Dynamic control with Eager
데이터셋에도 성능 향상이 있습니다.
Graph의 장점을 살려 Eager context 안에서도 빠르게 데이터셋을 순회할 수 있도록 
라이브러리가 최적화되었습니다. Eager가 프로토타이핑과 디버깅을 쉽게 해 주는 동시에
우리가 모델을 더 잘 볼 수 있도록 tf.keras는 보이지 않는 곳에서 Eager-friendly 함수들을 만들어 줍니다.
마지막으로 Tensorflow runtime이 유기적으로 성능 최적화와 확장성을 고려해 줍니다.
**또한 tf.keras의 모델 내부 구현도 Eager mode로 실행시키는 것이 가능합니다.**

```python
class Dynamic(tf.keras.layers.Layer):
  def call(self, inputs):
    if tf.reduce_max(inputs) < 10:
      inputs = inputs * 5
    return inputs

  model = tf.keras.models.Sequential([..., Dynamic(10), ...])

  model.compile(..., run_eagerly=True)
  model.fit(x_train, y_train, epochs=5)
```

이 예제에서 모델 컴파일`model.compile(..., run_eagerly=True)`시에 
인자로 `run_eagerly`를 명시했습니다. 이렇게 되면  모델 내부에서도 
Python control flow, Eager mode를 활용할 수 있게 됩니다.  
(이 인자는 성능 최적화를 위해 기본값이 False로 되어 있으나, 값을 주면
Keras Model 안에서도 Eager mode처럼 특정 레이어를 지난 뒤의 값처럼 
모델 내부에서 사용되는 변수도 찍어서 바로 확인할 수
있게 됩니다. 더 자세한 내용은 [Tensorflow tf.keras.model API](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#run_eagerly)를 확인해 주세요.)

### Consolidation under Keras
 ![image](https://user-images.githubusercontent.com/25409073/54267871-5d6beb80-45bd-11e9-9f3c-db63d6f43bfb.png)
 2.0에서의 큰 변화들 중 하나는, **Keras를 중심으로** Tensorflow의 주요한 API들이 병합되었다는 것입니다.
 중복되는 클래스들을 삭제하고, 어떤 클래스를 사용해야 하는지, 언제 사용해야 하는지를
 알기 쉽도록 했습니다.
 

### tf.keras.optimizer.\*
```python
# Optimizer example
tf.keras.optimizers.{ Adadelta, Adagrad, Adam, 
                      Adamax, Nadam, RMSProp, SGD }

optimizer = tf.keras.optimizers.Adadelta(clipvalue=0.)

# Hyperparameters are settable attributes
optimizer.learning_rate = .3

# Fully serializable
config = optimizer.get_config()
optimizer2 = tf.kears.optimizers.Adadelta.from_config(config)
```
이제 Tensorflow에는 위처럼 단 한 세트의 Optimizer들만 남겨집니다. Eager mode이든 
아니든, 한 대의 머신이든 아니면 분산된 훈련 환경이든지 상관없이 Tensorflow의 모든 곳에 사용가능합니다. 
이제 Python attribute를 다루듯이 Optimizer의 하이퍼파라미터를 설정할 수 있으며(`optimizer.learning_rate = .3`)
네트워크의 가중치값과 Optimizer 설정값들을 Tensorflow의 `checkpoints` 포맷, 혹은
Keras backend에서 사용하는 포맷으로 저장할 수 있습니다.


### tf.keras.metrics.\*
```python
# Example subclassing Metrics
class Lottery(tf.keras.metrics.Metrics):
  def __init__(self, magic_numbers):
    self.magic_numbers = self.add_weight(magic_numbers)
  ...

model.compile(
  'sgd',
  metrics=[Lottery([34, 11, 4, 20, 19, 85]),
           tf.keras.metrics.TopKCategoricalAccuracy(k=5),
           tf.keras.metrics.TruePositives()])
```
Evaluation을 위한 Metrics 또한 마찬가지로, 이전의 Tensorflow와 Keras를 
모두 아우르는 한 세트의 metrics들만 남겨졌으며 Subclassing을 통해 커스터마이징이 가능합니다.

### tf.keras.losses.\*
```python
# Example subclassing Loss
class AllIsLost(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.math.equal(y_pred, y_true)

model.compile(
  'sgd',
  losses=[AllIsLost(),
          'mse',
          tf.keras.losses.Huber(delta=1.5)])
```
Loss 또한 가장 많이 쓰이는 loss들을 built-in으로 제공하는 동시에, 위의 코드처럼 
클래스를 상속받아 임의의 loss를 만드는 것도 가능합니다.

### tf.keras.layers.\*
```python
item_input = tf.keras.layers.Input(tensor=items, name='item_in')

embedding_item = tf.keras.layers.Embedding(
  num_items, mf_dim + model_layers[0] // 2,
  embeddings_initializer=embeddings_initializer,
  input_length=1, name='embedding_item')(item_input)

pair_vector = tf.keras.layers.concatenate(
  [embedding_user, embedding_item]
```
마지막으로 keras 스타일로 새롭게 추가된, 혹은 keras에서 그대로 넘어온 built-in layer들입니다.
이들 역시 Class 기반이며 자유롭게 커스터마이징할 수 있습니다.

### RNN layers
RNN 계열의 Layer들은 Tensorflow 2.0부터 조금 특별해집니다.
```python
# TF1.x: RNN layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10)

if tf.test_is_gpu_available():
  model.add(tf.keras.layers.CudnnLSTM(32))
else:
  model.add(tf.keras.layer.LSTM(32))
```
Tensorflow 1.x에서 LSTM과 GRU는 각각 여러 버전의 LSTM, 여러 버전의 GRU가 존재했습니다.
사용자가 어떤 device를 사용하고 있느냐에 따라 최적의 퍼포먼스를 낼 수 있는
별도의 Layer들이 따로 존재했고 이를 모델을 훈련하기에 앞서 사용자가 
직접 최적의 Layer를 선택해주어야 했습니다.
```python
# TF2.0: RNN layers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10)

# This will use a Cudnn kernel when a GPU is available, otherwise calls basic LSTM()
model.add(tf.kears.layer.LSTM(32))
```
Tensorflow 2.0에서는 단 하나의 LSTM layer와, 단 하나의 GRU layer만 존재합니다.
사용자가 device와 관련한 것들을 알 필요가 없도록 
런타임에 Layer가 최적의 버전을 선택합니다. 위 코드에서는 사용가능한 GPU가 
있다면 Cudnn 버전의 LSTM을 호출할 것이고, 그렇지 않다면 CPU 버전의 LSTM layer를 
호출하게 됩니다.

### Customizable Layer
```python
class Flip(tf.keras.layers.Layer):
  def __init__(self, pivot=0, **kwargs):
    super(Flip, self).__init__(**kwargs)
    self.pivot = pivot

  def call(self, inputs):
    return self.pivot - inputs

x = tf.keras.layers.Dense(units=10)(x_train)
x = Flip(pivot=100)(x)
```
 Custom layer의 예시입니다. `tf.keras.layers.Layer`를 상속받고
`__init__()`, `call()` 두개의 함수를 구현하면 됩니다. 

### Tensorflow Addons
![image](https://user-images.githubusercontent.com/25409073/54325452-9b145700-4645-11e9-81b7-0c4fd4643856.png)

Tensorflow 커뮤니티 레포지토리인
[**tensorflow/addons**](https://github.com/tensorflow/addons)에서는 복잡한 custom layer를 포함해서
Tensorflow의 여러 base 클래스를 상속받아 구현된 여러 **실험적인** 커스텀 모듈(custom layers, metrics, losses ..)들을 
모아놓은 레포지토리입니다. 이들을 쉽게 가져와 바로 사용하는 것이 가능합니다.

### Keras Integration
Tensorflow 2.0을 만들면서 가장 먼저 한 일이 Tensorflow의 API들을 간소화하고, 또 유기적으로 연결하는 것이었고,
그 다음 **Keras를 중심으로** 기존 Tensorflow의 모든 기능들을 통합하는 것이었습니다.

### Keras Integration - tf.feature_column
기존 `tf.estimator`의 아주 큰 강점 중 하나는 **'Easily configurable structured data'**였습니다. 
즉 다양한 형태의 데이터를 파싱해주는 재사용 가능한 data pipeline을 `tf.feature_column` API 를 활용해 
아주 쉽게 설계할 수 있었습니다. 
```python
# Structured data parsing example using 'tf.feature_column'
user_id = tf.feature_column.categorical_column_with_identity(
                'user_id', num_buckets=10000)
uid_embedding = tf.feature_column.embedding_column(user_id, 10)

# 3 columns that will feed into keras model
columns = [uid_embedding,
           tf.feature_column.numeric_column('visits'),
           tf.feature_column.numeric_column('clicks')]

feature_layer = tf.keras.layers.DenseFeatures(columns)

model = tf.keras.models.Sequential([feature_layer, ...])
```
이제 Tensorflow 2.0에서는 이 `tf.feature_column` API가 `tf.estimator` 뿐만 아니라 
위 코드처럼 `tf.keras`의 model에도 호환가능합니다.

### Keras Integration - Tensorboard
이제 모델에 데이터를 넣고 훈련할 준비가 되었습니다. 모델 훈련 과정에서
가장 많이 사랑받았던 Tensorflow의 툴 중 하나는 바로 **Tensorboard**입니다.
```python
tb_callback = tf.keras.callbacks.Tensorboard(log_dir=log_dir)

model.fit(
  x_train, y_train, epochs=5,
  validation_data=[x_test, y_test],
  callbacks=[tb_callback])
```

2.0에서 `tf.keras`에서의 Tensorboard 시각화는 단 한줄이면 충분합니다.
위 코드처럼 keras model 훈련시에 **Tensorboard callback**을 추가하기만 하면,

![image](https://user-images.githubusercontent.com/25409073/54328396-14657700-4651-11e9-8617-f07c0610b41c.png)  
그림처럼 우리가 설계한 모델의 Training 과정 중 accuracy, loss값, Layer마다의 그래프 구조도 확인할 수 있게 됩니다.
<br />

또 한 가지가 더 있습니다.  

![image](https://user-images.githubusercontent.com/25409073/54328478-66a69800-4651-11e9-8844-ca4c57239586.png)  
 모델의 **profile**을 분석할 수 있게 됩니다.  **profile** 탭에는
모델이 어떤 device 위에서 연산이 실행되는지를 알려주는 device placement와, performance 정보를 더 
자세히 살펴볼 수 있어 사용자가 데이터 파이프라인의 bottleneck을 최소화하는 방법을 찾는데 도움을 줄 수 있습니다.

### Going big: Multi-GPU
`tf.distribute.Strategy` API는 모델의 Training workflow를 분산시켜 처리하기 위해 만들어졌습니다.
 이 API는 기존 1.x에서도 존재했으나 Keras와 잘 호환되고 쉽게 사용할 수 있게, 또 굉장히 다양한 분산 환경을 
 지원하는 방향으로 재설계되었습니다. 역시 `MirroredStrategy()`처럼 바로 쓸 수 있는 built-in set을 제공합니다.
```python
# MirroredStrategy example:
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=[10]),
    tf.kears.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(...)
  ...
```
API는 **strategy의 scope를 정의해 사용할 수 있습니다**. 위 코드에서는 `strategy`객체를 만든 뒤 
scope를 정의해 그 안에 keras model을 컴파일하고 훈련시켜 Multi-gpu 환경에서의 분산 처리를 가능하도록 합니다.  

keras 환경 안에서 이 distribution strategy API를 통합했기 때문에 단 몇 줄 만으로 많은 것을 할 수 있습니다.
데이터를 batch 단위로 미리 불러오고(prefetch), `AllReduce`를 사용해 내가 정의한 변수들이 지금 사용가능한 모든 
device들이 참조할 수 있게 함으로써 90% 이상의 분산 처리 효율을 달성할 수 있었습니다.

즉, 이제 우리는 코드를 변경하거나 Keras의 편의성을 포기하지 않고도 단 몇 줄의 코드만으로
모델의 속도를 높일 수 있게 되었습니다.

### To SavedModel and beyond

이제 모델을 트레이닝했으니 다른 시스템, 모바일 기기, 또는 다른 프로그래밍 언어 환경에서의
모델 배포를 위해 패키징을 할 차례입니다.
```python
saved_model_path = tf.keras.experimental.export_saved_model(
  model, '/path/to/model')
new_model = tf.keras.experimental.load_from_saved_model(
  saved_model_path)

new_model.summary()
```
이제 Keras model은 모든 Tensorflow ecosystem에서 작동하는 
serialization format인 `saved_model`로 바로 추출이 가능합니다.
이 기능은 지금 Alpha version에서는 아직 완벽하지 않습니다만 곧 TF Serving이나 TF Lite 등에서
바로 사용할 수 있는 모델을 Keras에서  바로 추출할 수 있게 될 것입니다.
또 당연하게도 모델을 다시 불러와 재훈련시키거나 다른 작업을 하는 것도 가능합니다.

### Coming soon: Multi-node synchronous
여기까지 Tensorflow 2.0 Alpha version에서 추가되거나 바뀐 내용들이고, 마지막으로
이제 몇 달 안에 곧 추가될 기능들을 간략히 소개해 드리겠습니다.

```python
# MultiWorkerMirroredStrategy example:
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  model = tf.keras.models.Sequential([
    ...
  ])
  model.compile(...)
  model.fit(...)
  ...
```
이전에 소개해드린 `MirroredStrategy()`는 하나의 머신 안에서 여러 Device들이 있는
환경을 가정하고 설계된 API입니다. (한 대의 머신에 꼽혀있는 여러 GPU 카드를 모두 활용하고자 할 때)
역시 똑같은 Keras model에서 `MultiWorkerMirroredStrategy()`를 사용하면 모델 연산을 여러 개의 
노드, 즉 여러 대의 머신에 분산시켜 처리하게 됩니다. 이 API는 아직 개발중이며,
nightly 버전에서는 지금도 사용가능합니다. 그리고 다음 release에서는 
Colab, 혹은 Google cloud 바탕의 Multiple TPU에서의
분산 처리를 위한 API 또한 곧 나올 것으로 기대하고 있습니다.

### Onwards and upwards
![image](https://user-images.githubusercontent.com/25409073/54406928-6379dd80-471f-11e9-9521-e23edf0947f8.png)

그리고 2.0 final release를 향해 개발하면서 계속해서 keras에게 확장성을 부여하고 있습니다.
현재 tf.estimator에서 지원되고 있는 비동기 방식의 training strategy를 `ParameterServerStretegy`라는 이름으로
Keras에 통합할 예정이고, High-level API보다 더 높은 수준의 API를 위해 **Canned Estimator** API 또한 Keras API로 가져갈 것입니다.
또 Google에서 사용되는 모델처럼 굉장히 거대한 모델은, 
**모델 안의 variable들을 파티셔닝**해 여러 머신에서 처리할 수 있도록 할 예정입니다.










