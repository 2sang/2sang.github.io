---
layout: post_without_img
title: "[WIP]2019 TF dev summit(2) - Introducing Tensorflow 2.0 and its high-level APIs"
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
### Painful moments
Tensorflow는 가끔씩 사용하기 고통스러웠고, 또 어려웠습니다.
![image](https://user-images.githubusercontent.com/25409073/54179727-fecf4080-44dc-11e9-9e1a-0989133cde8b.png)  

**Session**을 사용하는 것은 평범한 파이썬 유저인 우리들에게 자연스럽지는 않았고,
또 Tensorflow가 성장하면서 계속해서 라이브러리가 복잡해지고 헷깔리기 시작했습니다. 
Tensorflow로 아주 많은 것을 할 수 있었지만, Tensorflow를 **어떻게** 사용하는 것이
가장 잘 사용하는 것인지는 확실하지 않았습니다.

![image](https://user-images.githubusercontent.com/25409073/54178470-448a0a00-44d9-11e9-92ce-b942ff8d0141.png)  

그렇게 Tensorflow가 발전하면서 저희도 정말 많이 배웠습니다. 사용자 입장에서의 빠른 프로토타이핑, 
그리고 쉬운 디버깅이 절실하게 필요하다는 것을 느꼈습니다. 

### Install
![image](https://user-images.githubusercontent.com/25409073/54192070-7f506a00-44fa-11e9-9d0c-856755cf060e.png)  

많은 것들이 보완된 Tensorflow 2.0 Alpha version을 릴리즈했고, 이 커맨드로 바로 설치해 실행할 수 있습니다.
그래서, 어떤 점이 바뀌었을까요?  
<br />

### TF 2.0 - Usability
![image](https://user-images.githubusercontent.com/25409073/54191061-7eb6d400-44f8-11e9-9643-717f8395cf33.png)  
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

### TF 2.0 - Clarity
![image](https://user-images.githubusercontent.com/25409073/54184152-45c33300-44e9-11e9-8155-0a4cd22db000.png)  

Tensorflow 2.0은 Major release이므로, 이제 버릴 것들은 버려야 했습니다. 중복된 기능들을 삭제했고 API들을 통합했습니다.
API가 사용자들에게 더 일관성있게 보일 수 있도록 신경썼으며, Tensorflow 뿐만 아니라 Tensorflow와 함께 성장하고 있던 
모든 Ecosystem들에게도 이런 변화들이 적용되었습니다. 모든 Ecosystem에서 추가적인 변환작업이 없이 통용될 수 있는 
모델의 **Exchange format**도 정의했습니다.  
<br />

### TF 2.0 - Flexibility
![image](https://user-images.githubusercontent.com/25409073/54191123-9beba280-44f8-11e9-9f71-f63202c65ffe.png)  

그렇게 2.0 릴리즈에서 많은 것들을 버리게 되면서 Tensorflow는 훨씬 더 유연해집니다.
low-level API, `tf.raw_ops`가 Tensorflow의 모든 내부 연산들을 조작할 수 있게 해 줄 것이고,
`variables`, `checkpoints`, `layers`, 이외의 많은 Tensorflow 주요 컨셉들의 클래스를 상속받는 방법으로
각 컨셉의 Custom class를 만드는 것이 가능합니다.

### Upgrade to TF 2.0
![image](https://user-images.githubusercontent.com/25409073/54191307-fdac0c80-44f8-11e9-9662-9f2f9a9cd366.png)
그렇다면 이제 우리의 질문은 _'어떻게 업그레이드를 하나'_ 가 됩니다.
어떤 버전 업그레이드라도 그 과정이 쉽지 않음을 우린 잘 알고 있고, 특히 이런 Major upgrade의 경우 더욱 그렇습니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/54191551-7f9c3580-44f9-11e9-84c3-d4647c91c1fd.png)
- 사실, 최근 구글 또한 엄청난 규모의 코드베이스에 대해 TF2.0으로의 업그레이드를 시작했습니다.
이 업그레이드가 진행되는 동안 많은 양의 Migration guide, 그리고 Best practice들이 공개될 것이고 
우리가 구글에서 하고 있는 것처럼 여러분들 또한 충분히 할 수 있을 것이라고 생각합니다.
- 또 2.0으로의 마이그레이션을 위해 굉장히 많은 툴들을 제공합니다. 가장 먼저
Tensorflow 2.0의 일부분으로써 `tf.compat.v1` 모듈이 탑재됩니다. 2.0에서 사용할 수 없게 된 함수를
꼭 사용해야 하는 경우라면 이 패키지에서 이전에 사용했던 함수를 찾을 수 있습니다.
**하지만, `tf.contrib` 패키지는 포함되지 않습니다.**
- 그리고, Tensorflow 1.x으로 작성한 스크립트 파일을 바로 변환시켜주는 `tf_upgrade_v2` 커맨드 라인 툴을 사용하면
2.0에서 바뀐 함수명으로의 변경이나 함수 Argument 순서의 변경 등 버전 변경사항을 적용해 줍니다.
이 때, 2.0에서부터 삭제되거나 Deprecated된 함수를 사용하고 있었다면 위에서 설명한 `tf.compat.v1`패키지의 함수로 변환해 주고,
`tf.contrib` 모듈을 사용했던 경우라면 변환이 실패할 수 있습니다.  
<br />

### tf_upgrade_v2
![image](https://user-images.githubusercontent.com/25409073/54195977-71531700-4503-11e9-99db-9fe8e4383778.png)  


```bash
# Example file conversion in command line
$ tf_upgrade_v2 --infile cnn_model.py --outfile cnn_model_upgraded.py
```
위와 같이 변환 결과에 대한 정보를 커맨드 라인에서 볼 수 있습니다.
주의할 점은 이 자동변환 툴 `tf_upgrade_v2`는 1.x버전의 코드를 2.0에서도 문제없이 실행될 수 있도록 고쳐주지만,
코딩 스타일을 새로운 Tensorflow 2.0의 스타일로 고쳐주지는 않는다는 점입니다.

## High level APIs in Tensorflow 2.0 - Karmel Allison
### High-level API
![image](https://user-images.githubusercontent.com/25409073/54201052-196edd00-4510-11e9-9a05-fc7cc3589ea0.png)  

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
Keras는 그 자체로 Pythonic하고 또 배우기 쉽도록 만들어져 사람들을 Machine learning의 세계로 더 쉽게 끌어당길 수 있습니다.
위 코드는 전체 모델의 완전한 구조와 Training loop를 보여줍니다. 
이렇게 사용자들이 장황한 코드를 작성할 필요 없이 쉽게 구조를 바꾸거나 모델을 다시 만들 수 있습니다.

```python
```










