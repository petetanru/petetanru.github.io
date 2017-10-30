---
layout: post
title:  "Neural Machine Translation with Thai"
date:   2017-10-20 14:00:00 +0700
tag: AI, rnn, CNN, deep-learning, NLP, NMT
---

This post is still a work in progress. Will update as I try out more stuff

Link to [repository](https://github.com/petetanru)

- Updated 30 Oct 2017 
  - Performs BLEU measuring against validation set
  - Encoder for character, character-clusters, sub-word/BPE, and word level. 

## Current Challenges

Machine translation with Thai has historically not been good. Several of the key challenges include:

1. **Word segmentation** - Thai does not use space and word segmentation not easy. It boils down to understanding the context and ruling out words that do not make sense.

   ```
   Example 1: ตากลม could either be ตา-กลม (round eyes) or ตาก-ลม (drying by wind). 
   Example 2: คนขับรถ (driver) is made of three words คน-ขับ-รถ(person, drive, car) 
   but this similar pattern of noun-verb-noun does not always make up a word, such 
   as คนโขมยของ (person, steal, stuff). 
   ```

2. **Start/End of sentence marking** - This is arguably the biggest problem for the field of Thai Machine Translation. The lack of end of sentence (EOS) marking makes it hard to create parallel corpuses, so that further research could be conducted. The root of the problem itself is two-pronged. In terms of writing system, Thai uses space to indicate both commass and periods. No letter indicates an end of a sentence. In terms of language use, Thais have a habit of starting their sentences with connector terms such as 'because', 'but', 'following', etc, making it hard even for natives to tell where the end of sentence is when translating.

   i.e.: "should this connector or the next connector be the end of a sentence?"

   A simple way of handling this is to treat them all as EOS, but that would obviously create several short sentences that may be is unnaturally lacking in context. Again, the point is that no fixed rules would work here, only probabilities. Without proper editing, it is not uncommon to see run-on sentences when translating Thai writings to other languages.

3. **Lack of large parallel data** - This is probably self explanatory.. But most recently, we now have TED Talks from [WIT3!](https://wit3.fbk.eu/)

The lack of a perfect (to trivialize, perfect here only means something that spacing and periods achieve in English) word segmenting (#1) and sentence boundary marking (#2) tools create problems that snowball into the performance of higher level task, such as translation.  

## Addressing the challenges

### Vocabs ###

The most obvious input that you would put into your NMT would be words, but the state of art models have found great success in using subword units and character levels as well! Given that Thai has no perfect tokenizer, it would be interesting to see if the model could learn to form words on teh fly and perform well! 

Namely, I will be evaluating the following ways to capture vocabs:

1. **Word level** - This will be our baseline. Rakpong recently made a CNN-based tokenizer that performs quite adequately, achieving F1 of 98.1%, only a bit lower than NECTEC's private F1 98.6% tokernizer.

2. **Character level** - Traditionally, character level RNNs for translation tasks were not very popular because the overly long sequence weould create vanishing gradients problem, and it would also make the model too computationally expensive. Recently, Lee et al. (2017) proposed a character level NMT that does address the long sequence problem by utilizing 1D CNNs to create different sized n-grams nodes, and compress the sequence with maxpool striding.

3. **Byte-Pair Encoding / Wordpiece** - Sennrich et al. (2016) and Wu et al. (2016) proposed a way to represent language by breaking words down to subword units, and using the most common 8k - 32k of those subword word units as your vocab.

4. **Thai Character Cluster** - Theeramunkong et al. (2000) suggested a technique called 'Thai character clustering' (TCC) that groups Thai characters based on the Thai writing system into clusters that cannot be further separated. This is possible because in Thai, there are vowel and tone marks that cannot stand alone. This is similar to BPE/wordpiece, but rule based rather than data-driven. The english equivalent would be to call 'qu' a character cluster, since 'u' always follow 'q'.

    > Although commonly referred to as the "Thai alphabet", the script is in fact not a true alphabet but an abugida, a writing system in which each consonant may invoke an inherent vowel sound. In the case of the Thai script this is an implied 'a' or 'o'. Consonants are written horizontally from left to right, with vowels arranged above, below, to the left, or to the right of the corresponding consonant, or in a combination of positions. - Wikipedia

### Multi-lingual ###

Is meaning language bound, or do concepts exist in abstract which then get decoded to languages? For the machine learning field, the answer seems to be the latter, as performance can often be incrased by training on multiple source languages, and translating to one. 

The focus has mostly been on non-Asian languages though, especially those that share similar alphabets. I want to see whether SEA languages can learn from each other, or how Asian2Asian languages perform.

Here's my TODO list.

- [x] TH-EN
- [ ] TH-VI
- [ ] TH-KR
- [ ] TH+VI - EN
- [ ] TH+VI - KR
- [ ] TH+KR - EN

I focus on translating with Thai as the source language, rather than the target, because the current [Moses BLEU script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) is built for languages that does not use spaces in an ambiguous way (ie: Thai). 

## Model and hyper parameters ##

My models are (very) simplified version of actual state-of-art NMTs. For the encoder, I use either a typical GRU or Lee's implementation of [CNN+GRU](https://arxiv.org/abs/1610.03017) for character level encoding. For the decoder, I use [Luong's](http://aclweb.org/anthology/D15-1166) global attention GRU model. The default parameters are: 

|                     | Default parameters                       |
| ------------------- | ---------------------------------------- |
| Hidden units        | 256                                      |
| Layers              | 1                                        |
| Direction           | 1                                        |
| Embedding size      | 256 for words and TCC<br />128 for characters |
| K width and filters | [1,2,3,4,5,6,7,8] with<br />[200,200,250,250,300,300,350,350] |
| Learning rate       | 1e-4                                     |
| Optimizer           | Adam                                     |

## Preprocessing ##

Each sentence is convereted to a vector with the maximum length of the longest sentence in the mini batch. The data that I use is are 'segments of words', which is basically a line from TED Talk's subtitle. They are not necessarily a sentence. This carries some misalignment risk, but is much more viable for our task since the length is much shorter. 

## Result of comparison ##

| Model      | Base line NMT | TCC      | BPE  | Char2word  |
| ---------- | :------------ | -------- | ---- | ---------- |
| Input      | Word          | TCC      |      | Characters |
| Vocab size | 28701         | 2737     |      | 87         |
| Encoder    | GRU           | GRU      |      | CNN-GRU    |
| Decoder    | GRU-attn      | GRU-attn |      | GRU-attn   |
| BLEU Score | 10.7          | 10.3     |      | 7.7        |

As reference, our BLEU scores seem to be on the right track. The baseline BLEU score of 10.7 is about the same as this [paper's](https://arxiv.org/pdf/1606.07947.pdf) small model baseline for TH-EN, which used the attention LSTM with 100 hidden units (they also tried 500 units) and beam search k=1 to 10.6 BLEU on the 2015 TED Talk corpus. 

## Analysis ##

**TCC**: I'm suprised by how well TCC performed, achieving only 0.4 lower BLEU despite the vocab size being only 1/10th of the word level model. Google reports that it gets good results with 8k - 40k vocab size when using subword units. TCC produces 1/4th the size of vocab yet still manages to learn quite well.

**Character**: I know that Lee's paper manage to reach state of art, using english-german corpus. I'm surprised this has not worked as well for TH-EN. I'd like to see whether its performance would start to catch up given a large enough network though.


-[ ] TODO: compare TCC to BPE. 

## Quick References ##

1. A lot of ideas and design taken from [Lee's NYU code](https://github.com/nyu-dl/dl4mt-c2c). Particularly preprocessing. My implementation of his CNN-GRU model MIGHT have some mistakes though, since I don't code Theano and base my pytorch implementation mostly from reading his [paper](https://arxiv.org/abs/1610.03017).
2. TED Talk data from [WIT3](https://wit3.fbk.eu/). 
3. TCC implemented by [Korakot](https://github.com/korakot). 
4. Deepcut Thai tokenizer from [Rakpong](https://github.com/rkcosmos/deepcut)
5. Global attention from Luong [paper](http://aclweb.org/anthology/D15-1166). 
6. Moses for Tokernizer and BLEU script. [Link](https://github.com/moses-smt/mosesdecoder)
7. Academia, stackoverflow, google, and the internet for existing, and making it possible for a self-taught person like me to put something like this together.  

