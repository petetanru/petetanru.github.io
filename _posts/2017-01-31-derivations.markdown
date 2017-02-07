---
layout: post
title:  "Derivations from ML course"
date:   2017-01-31 16:32:00 +0700
tag: math, AI
---

I'm current going through CS224d, Natural Language Processing, and the assignment involves several derivations. I want to log some of my work here especially because many solutions over the internet gloss over basic steps and are not as friendly for newcomers. This will hopefully also help me when I forget hwo to do them.

## Deriving Softmax w.r.t. to its input

$$ f(x_j) = softmax(x_j) = \frac{ \exp^{x_j} }{ \sum_{k}^{n} \exp^{x_k} } $$

Using quotient rule where:

$$ f(x) = \frac{g(x)}{h(x)} $$ 

$$  f'(x) = \frac{ g'(x)h(x) - g(x)h'(x) }{ h(x)^2 } $$

​	

When i = j:

$$ \frac {\partial g(x_j)}{\partial x_i} = \exp^{x_i} $$

$$ \frac {\partial h(x_j)}{\partial x_i} = \sum 0 + 0 + 0 + 0 + \exp^{x_{i=j}} + 0 + 0  = \exp^{x_i} $$


So:

$$ \frac {\partial f(x_j)}{\partial x_i} = \frac{ \exp^{x_i} * \sum \exp^{x_k} - \exp^{x_j} * \exp^{x_i} }{ (\sum\exp^{x_k})^{2} }   $$

$$ = \frac{\exp^{x_i}}{\sum\exp^{x_k}} * \frac{  \sum \exp^{x_k} - \exp^{x_j} }{ \sum\exp^{x_k}}  $$

$$ = f(x^j) * (\frac{\sum\exp^{x_k}}{\sum\exp^{x_k}} - \frac{\exp^{x_i}}{\sum\exp^{x_k}} ) $$

$$ = f(x^j) * (1 - f(x^j))  $$



When $i \neq j$, since $\exp{x_i}$ is not a function of $\exp{x_j}$:

$ \frac {\partial g(x)}{\partial x_i} = 0  $  and  $ \frac {\partial h(x)}{\partial x_i} = \exp^{x_i }  $  because $i$ is some number in $\sum \exp{x_k}$

Using quotient rule: 

$$ \frac {\partial f(x_j)}{\partial x_i} = \frac{ 0 - \exp^{x_j} * \exp^{x_i} }{ (\sum\exp^{x_k})^{2} }   $$

$$ = - \frac{\exp^{x_j}}{\sum\exp^{x_k}} * \frac{\exp^{x_i}}{\sum\exp^{x_k}}$$

$$ = - f(x_j) * f(x_i)$$



To summarize: 

Given $p_j = \frac{e^x_j}{\sum_k e^x_k} $

$$\frac{\partial p_j}{\partial \theta_i} = p_i (1 - p_i), i =j$$  

$$\frac{\partial p_j}{\partial \theta_i} = -p_i p_j, i \neq j$$



# Deriving Cross Entropy w.r.t Softmax's input

$CE = - \sum_j{y}\log(\hat{y})$  where  $\hat{y} = softmax(\theta) $  and  $y$ is a one-hot vector

With chain rule: 

$$f'(x) = \frac{\partial f(x)}{\partial g(x)} * \frac{\partial g(x)}{\partial h(x)}$$

$$ \frac{\partial CE}{\partial\theta_i} = -\sum \frac{ \partial y \log(\hat{y})}{\partial \theta_i}$$

$$= -\sum_j y_j * \frac{1}{\hat{y}_j}\frac{ \partial\hat{y}_j}{\partial \theta_i}$$



When $i = j$ 

$$-\sum_j y_j * \frac{1}{\hat{y}_j} * \frac{ \partial\hat{y_j}}{\partial \theta_i} = - y_i * \frac{1}{\hat{y}_i} *  \hat{y}_i (1 - \hat{y}_i)$$

$$ = -y_i (1-\hat{y}_i)$$



When $ i \neq j $ : 

$$ -\sum y_j * \frac{1}{\hat{y}_j} * \frac{ \partial\hat{y}}{\partial \theta_i} =  -\sum y_{i \neq j} * \frac{1}{\hat{y}_j} * (- \hat{y}_i \hat{y}_j) $$

$$= \sum_{i \neq j}  y_i\hat{y}_i$$



Combining the two

$$-\sum y_j * \frac{1}{\hat{y}_j}\frac{ \partial\hat{y}}{\partial \theta_i} = -y_i (1-\hat{y}_i) +  \sum_{i \neq j} y_i \hat{y}_i $$

$$= y_i \hat{y}_i - y_i +  \sum_{i \neq j} y_i\hat{y}_i $$

$$ = \hat{y}_i \sum_j y_i - y_i $$

We know that sum of $y_i$ is 1, so the solution is:

$$ = \hat{y}_i - y_i$$

or equivalently: 	

$\hat{y}_i - 1,$  	$i = j $

$\hat{y}_i,$  		$i \neq j$



