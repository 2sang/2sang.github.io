---
layout: post_without_img
title: "2019 TF dev summit - Airbnb uses ML to help categorize its listing photos"
author: "Sangsu Lee"
categories: journal
tags: [documentation,sample]
image: tf-logo.png
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/b5Rs1ToD9aI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

2019 Tensorflow dev summit

## Docker
Docker는, 

## 사실 문제는,
햄버거 그림 here
정말 문제는 개발환경 셋업시 충돌이 나게 되면 **어떤 부분이 잘못되었는지 파악을 하기 힘들다는 것입니다.**  
잘못된 부분을 알면 그 부분만 고치면 되지만, 우리가 직면하는 대부분의 패키지 충돌 / 개발 스택들의 버전 충돌은 
어디가 잘못되었는지 모르는 경우가 많죠.  결국엔 다 엎어버리고 OS부터 다시 시작하는게 낫겠다는 생각이 들기도 하구요. ㅠ.ㅜ
그러면 어떻게 해야 할까요? 

## Virtual environment tools
로컬에 pip, pip3 등으로 직접 패키지를 깔기보다 **패키지를 관리해 주는 관리 툴**을 사용하는 것도
한가지 방법이 될 수 있습니다. 이렇게 할 경우에 프로젝트별로 사용되는 패키지들을 독립적으로 분리시킬 수 있는데요,
파이썬 내장 툴 `venv`, 혹은 data science 프로젝트들에 특화된 `conda`(Anaconda)가
아마 한번쯤 들어보셨을 대표적인 파이썬 프로젝트 패키지 관리 툴들입니다.  
이런 툴을 사용해 패키지를 관리한다면, 충돌이 나더라도 '여기까지는 확실히 잘 되는' 버전들 위에 새로 다시 설치해 보거나,
아니면 최악의 경우 처음부터 시작하더라도 가상환경 이전의 환경만큼은 깨끗하다고 확신할 수 있으므로
OS를 재설치하는 시간을 아낄 수 있게 됩니다. 패키지 관리 툴의 동작 방식은 조금씩 다르지만 이들의 역할은 기본적으로 **프로젝트마다 가상 환경**을 만들어 주고 그 위에 파이썬 패키지를 설치하거나 삭제하는 등
**독립적으로** 패키지 관리를 할 수 있도록 도와줍니다.

## Docker로 딥러닝 개발환경을?
Docker의 고립된 컨테이너 안에서 프로젝트를 관리하는 것 또한 위에서 설명한 것과 같은 원리로 보시되,  
**Tensorflow official docker image**를 사용해 딥러닝 환경을 구축하게 되면 [컨테이너의 OS - python - tensorflow]까지의
아주 잘 구축되어 있는 상태에서 그 위에 내가 원하는 환경 셋업을 시작할 수 있게 됩니다.(cpu 버전의 경우)  

특히, 환경 구축이 까다롭기로 악명높은 **tensorflow-gpu**의 경우 또한 [CUDA - cuDNN - tensorflow-gpu] 
의 안정적인 조합들 중 하나를 바로 받아와 실행할 수 있으며 공식 이미지이므로 이들의 버전 조합 사이에는 절대 충돌이 있을 수가 없다고 확신할 수 있습니다.  
다만 도커가 설치된 Host OS에는 GPU 카드에 맞는 드라이버가 설치되어 있어야 하고, 드라이버와 저 셋의 조합 사이에는 호환이 되어야 합니다.
(tensorflow-gpu 이미지가 드라이버와 맞지 않는다면 다른 조합의 CUDA-cudnn-tfgpu 버전을 금방 다시 받아와 시도할 수도 있습니다.)


## 원격 개발환경 셋업 예시
<셋업그림>  


