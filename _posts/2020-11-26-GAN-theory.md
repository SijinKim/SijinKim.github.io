---
title: "GAN theory(1) - global optimum"
excerpt: "GAN의 minmax problem으로 global optimum point가 보장되는 이유"
categories: 
   - Deep Learning
   - Machine Learning
   - GAN
use_math: true
---

## GAN의 global optimum point 찾기


GAN의 generator는 minmax problem을 통해 타겟한 데이터의 분포를 학습해간다.  
Discriminator를 속일 정도로 데이터의 분포를 잘 학습하여 '진짜'같은 fake 이미지를 생성할 수 있는 Generator를 만든다는 것인데,  

오, 어떻게 저런 생각을 했을까!  

GAN의 학습 방법을 처음 접했을 때 내가 했던 생각이었다.

놀란 마음은 잠시 뒤로하고, 이번 기회에 GAN을 깊이있게 공부해보고자 한다.
우선, GAN의 속고 속이는 방법이 모델의 global opmima 수렴을 보장하는 접근법인지 공부해보자.
(참조: <http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html> ) 

### GAN의 value function (minmax problem):

  
$ \underset{G}{min}\ \underset{D}{max}\ V(G, D) = E_{x\sim p_{data}}[log D(x)] + E_{z\sim p_z}[log(1-D(G(z)))] $
---
Discriminator가 잘 학습된 경우:  

$ {log D(x) = log(1) = 0} $  

  $ log(1 - D(G(z))) = log(1 - 0) = 0 $


Generator가 잘 학습된 경우:  

 $ log D(x) = log(0) = -\infty $  

 $ log(1-D(G(z))) = log (1 - 1) = -\infty $  

따라서, GAN이 잘 학습된다면, Discriminator 입장에서는 value function의 값을 최대화(maximum value: 0)하고 Generator 입장에서는 value function의 값을 최소화(minimum value: $ -\infty $ )하는 방향으로 학습이 이뤄진다.


#### 1. Generator fix 가정 적용하기 
minmax problem을 통해 학습이 잘 진행중인 GAN 모델이 있다고 하자. 이때, Generator를 fix하게 되면 GAN의 minmax problem을 오직 Discriminator를 maximize하는 문제로 전환할 수 있다. 즉,

$ \underset{G}{min}\ \underset{D}{max}\ V(G, D) $  


$ = \underset{D}{max}\ V(G,D) $  


$ = \int_x{p_{data}(x)\ log(D(x))}dx + \int_z{p_{z}(z)\ log(1-D(G(z)))}dz $ 


$ = \int_x{p_{data}(x)\ log(D(x)) + p_{gen}(x)\ log(1-D(x))}dx $ 

> probability density function space 가정을 통해 value function에 expectation 계산 적용

이때, $p_{data}(x)$, $p_{gen}(x)$, $D(x)$를 각각 $a$, $b$, $y$ 라고 놓으면 다음과 같은 식을 유도할 수 있다.


$ = \int_x{a\ log(y) + b\ log(1-y)}dx$

Generator가 고정된 상태에서 $V(G, D)$의 최대값은 0 이므로, 미분을 통해 위 식이 0 이 되는 point에서 Discriminator는 최적의 상태가 된다.


$ [a\ log(y) + b\ log(1-y)]' = \frac{a}{y} - \frac{b}{1-y} = a(1-y) - by = 0 $

$ \therefore y = \frac{a}{a+b} = \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)}$


즉, Generator가 fix됐을 때, optimal Discriminator는 아래와 같이 정의된다.

$ D_{G}^{*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)} $


이를 초기 minmax problem에 대입하여 reformulate하면 다음과 같이 정리할 수 있다. 

$ \underset{D}{max}\ V(G,D) $

$ = E_{x\sim p_{data}}[log D_{G}^{\*}(x)] + E_{z\sim p_z}[log(1- D_{G}^{\*}(G(z)))] $

$ = E_{x\sim p_{data}}[log D_{G}^{\*}(x)] + E_{x\sim p_{gen}}[log(1- D_{G}^{\*}(x)] $

$ = E_{x\sim p_{data}}[log \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)}] + E_{x\sim p_{gen}}[log(1 - \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)})] $

$ p_{data}$ 와 $ p_{gen} $ 의 분포가 동일할 때(generator가 data 분포 완벽히 모사), $ \underset{D}{max}\ V(G,D) $ 의 global minimum value의 획득이 보장된다. 즉,

$ For\ p_{data} = p_{gen}, \ $
$ \underset{D}{max}\ V(G,D) $

$ = E_{x\sim p_{data}}[log \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)}] + E_{x\sim p_{gen}}[log(1 - \frac{p_{data}(x)}{p_{data}(x) + p_{gen}(x)})] $

$ = E_{x\sim p_{data}}[log \frac{1}{2}] + E_{x\sim p_{gen}}[log(1 - \frac{1}{2})]$
$ = -log2 -log2 = -log4 $

즉, optimal Discriminator 는 $p_{data} = p_{gen}$인 경우만을 유일한 해로 하여 $ -log4$의 global minimum을 획득한다는 것을 알 수 있다.


