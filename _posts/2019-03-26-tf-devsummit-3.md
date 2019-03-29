---
layout: post_without_img
title: "2019 TF dev summit(3) - Airbnb uses machine learning to help categorize its listing photos"
author: "Sangsu Lee"
categories: journal
tags: [documentation,sample]
image: tf-logo.png
---


<iframe width="560" height="315" src="https://www.youtube.com/embed/tPb2u9kwh2w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 2019 TF dev summit series
Tensorflow Dev Summit 2019 발표 영상들을 코드와 함께 요약하고 있습니다.
오역이나 의역, 빠진 내용들이 있을 수 있으며 더 자세한 컨퍼런스 정보는 [Tensorflow Dev Summit Event page](https://www.tensorflow.org/dev-summit) 에서 확인하실 수
있습니다.  

[01 - Keynote]({% post_url 2019-03-10-tf-devsummit-1 %})  
[02 - Introducing Tensorflow 2.0 and its high-level APIs]({% post_url 2019-03-12-tf-devsummit-2 %})  
**[03 - Airbnb uses machine learning to help categorize its listing photos]()**  
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

## Airbnb
![image](https://user-images.githubusercontent.com/25409073/54988013-f0167c80-4ff8-11e9-815b-34f49a8b7707.png)  

Airbnb는 세계 여러 곳의 도시에서 숙소를 중개하는 온라인 서비스입니다.
8만개가 넘는 도시에서 500만개가 넘는 다양한 숙소들을 중개하고 있으며 
숙박시설의 이미지 데이터로는 아주 거대한 양을 보유하게 되었는데요, 투숙객이 숙박시설을 선택할 때의
가장 큰 결정요소 또한 숙소를 미리 볼 수 있는 다양한 이미지들입니다.
하지만 많은 경우에 집주인은 여러 방들 중 몇 개 방 사진만 업로드하고, 다른 방들의 사진은 올리지 않게 됩니다.
웹사이트에서 글을 써 다른 방들을 설명할 수 있지만 친절하게 그렇게 해 주는 사람들이 사실 많지는 않습니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/55251532-2a4f7a80-5294-11e9-99a3-70a2bccaf302.png)  

이미지에 어떤 방 정보가 담겨있는지 정확히 알아낸 뒤에 사용자들에게 '잘' 알려주어야 했습니다.
문제는 방대한 이미지의 양이었습니다. 기존의 기술로 약 5억개가 넘는 이미지를 모두 처리하려면 
최소 몇 달 이상이 걸릴 것이 분명했습니다. 우리는 Tensorflow를 활용해 이 기간을 단 몇 일로 줄일 수 있었습니다.  
<br />

![image](https://user-images.githubusercontent.com/25409073/55252082-5b7c7a80-5295-11e9-8417-b51076dfe20c.png)  

또 Airbnb의 end-to-end 머신러닝 플랫폼 **[Bighead](https://databricks.com/session/bighead-airbnbs-end-to-end-machine-learning-platform)**는,
머신러닝을 도입하는데 초기 어려움을 겪고 있는 조직에
Model 구축, Feature engineering, Serving을 처리하는 Tensorflow 컴포넌트들을 묶어 일관성 있는 머신러닝 workflow를
제공합니다. 이 프레임워크를 Tensorflow의 Cross API, 분산 GPU 컴퓨팅 모듈을 가져와 구축할 수 있었습니다.

![image](https://user-images.githubusercontent.com/25409073/55252675-c2e6fa00-5296-11e9-90e7-c4137cc5ced4.png)  
그리고 어떤 머신러닝 툴을 사용하느냐에 앞서 어떤 모델을 사용하느냐를 생각하게 되는데,.
ResNet50은 State-of-the-art 모델들 중 하나로 Bighead의 기본 아키텍쳐로 들어가게 됩니다. 이를 통해
궁극적으로 몇백개부터 몇백만개의 이미지들을 아주 빠르게 훈련하고 배포하는 파이프라인을 구축할 수 있습니다.  

![image](https://user-images.githubusercontent.com/25409073/55253142-eb232880-5297-11e9-9d3c-7edd97983373.png)  
최종 목표는 집주인이 업로드한 몇 장의 이미지 셋으로부터 이 이미지들이 단순히 화장실인지, 혹은
차고인지를 나타내는 것을 분류하는것부터 시작해 이 수영장이 야외 수영장인지, '멋진' 거실인지를 
상세히 분류해 사용자들이 알 수 있게 하는 것입니다.
<br />

![image](https://user-images.githubusercontent.com/25409073/55253312-5967eb00-5298-11e9-9999-06c58fcb17b3.png)  
또 미래에는 방 이미지 안에서 여러 물체들을 검출해 사용자가 특정한 시설이나 방에 갖추어진 물건들을 사이트에서
확인할 수 있도록 더 발전시키는 방향 또한 생각하고 있습니다.
<br />

![image](https://user-images.githubusercontent.com/25409073/55253351-743a5f80-5298-11e9-96ea-f5d7fd3b4da8.png)  
여러분이 만약에 Airbnb가 제공하는 컨텐츠들을 좋아하고 있다면, 이유를 머신러닝에서 찾을 수 있습니다.
머신러닝은 지금도 회사의 어느 곳에서나 활용되고 있으며
Social ranking, 가격 예측, 숙소예약 예측 등이 바로 이 몇백개 머신러닝 모델들의 결과가 됩니다.
또 이와 같은 여러 프레임워크가 계속해서 발전하고 있기 때문에 고객들이 
앞으로도 더 나은 결정을 할 수 있게 될 것입니다.


