---
layout: post
title:  "Char-based NMT"
date:   2017-10-20 14:00:00 +0700
tag: AI, rnn, CNN, deep-learning, NLP, NMT
---

Translating Thai is challenging because of two big reasons. 

1. Segmenting word is difficult because there are no spaces. Many words are compound words, made up of 2-3 words pieced together. It boils down to the context. 
2. Sentence boundary is not clear. Spaces are used as both commas and periods in Thai, and there is no clear indication when a sentence ought to stop. 

Because of these challenges, perhaps a machine translation model that is most suited for the task is one that could develop its own segmentation as it trains. 

I implemented a pytorch version a of this paper: https://arxiv.org/abs/1610.03017 in the following link: https://github.com/petetanru/NMT_pytorch

I begin by training it on a simple ENG-FRE corpus and it's performance has been competitive compared to other models that rely on tokenizers. 