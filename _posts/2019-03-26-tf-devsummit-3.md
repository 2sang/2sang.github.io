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
웹사이트에서 글을 써 다른 방들을 설명할 수 있지만 친절하게 그렇게 해 주는 사람들이 많지는 않습니다.


## Challenges



