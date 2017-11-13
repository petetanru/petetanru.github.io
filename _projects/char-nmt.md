---
layout: post
title:  "Neural Machine Translation with Thai"
date:   2017-10-20 14:00:00 +0700
tag: AI, rnn, CNN, deep-learning, NLP, NMT
---

Link to [repository](https://github.com/petetanru)

- Updated 8 Nov 2017
  - Added multilingual with google tokenizers and subword tokenizers (BPE/TCC)
  - Added bilingual with Vietnamese


- Updated 30 Oct 2017 
  - Performs BLEU measuring against validation set
  - Encoder for character and word level. 



## Current Challenges

Machine translation with Thai has historically not been good. Several of the key challenges include:

1. **Word segmentation** - Thai does not use space and word segmentation is not easy. It boils down to understanding the context and ruling out words that do not make sense.

   ```
   Example 1: ตากลม could either be ตา-กลม (round eyes) or ตาก-ลม (drying by wind). 
   Example 2: คนขับรถ (driver) is made of three words คน-ขับ-รถ(person, drive, car) 
   but this similar pattern of noun-verb-noun does not always make up a word, such 
   as คนโขมยของ (person, steal, stuff). 
   ```
   This is the same issue that other Asian languages such as Japense and Chinese face. For languages with space, a similar but less extreme problem would be multi-word expressions, like 'pomme de terre'. 

2. **Start/End of sentence marking** - This is arguably the biggest problem for the field of Thai Machine Translation. The lack of end of sentence (EOS) marking makes it hard to create parallel corpuses, the basis of most research in this field. The root of the problem is two-pronged. In terms of writing system, Thai uses space to indicate both commas and periods. No letter indicates an end of a sentence. In terms of language use, Thais have a habit of starting their sentences with connector terms such as 'because', 'but', 'following', etc, making it hard even for natives to tell where the end of sentence is when translating.

   i.e.: "should this connector or the next connector be the end of a sentence?"

   A simple way of handling this is to treat them all as EOS, but that would obviously create several short sentences that may be is unnaturally lacking in context. The point is that no fixed rules would work here, only probabilities. Without proper editing, it is not uncommon to see run-on sentences when translating Thai writings to other languages.

3. **Lack of large parallel data** - This is probably self explanatory.. But most recently, we now have TED Talks from [WIT3!](https://wit3.fbk.eu/)

The lack of a perfect (to trivialize, perfect here only means something that spacing and periods achieve in English) word segmenting (#1) and sentence boundary marking (#2) tools create problems that snowball into the performance of higher level task, such as translation.  



## Addressing the challenges

### Vocabs ###

The most obvious input that you would put into your NMT would be words, but the state of art models have found great success in using subword units and character levels as well! Given that Thai has no perfect tokenizer, it would be interesting to see if the model could learn to form words on the fly or subword units that are more useful than the whole word (which may be wrongly created anyway)! 

Namely, I will be evaluating the following ways to capture vocabs:

1. **Word level** - Rakpong recently made a CNN-based tokenizer that performs quite adequately, achieving F1 of 98.1%, only a bit lower than NECTEC's private state of art tokenizer with F1 at 98.6%.

2. **Character level** - Traditionally, character level RNNs for translation tasks were not very popular because the overly long sequence weould create vanishing gradients problem, and it would also make the model too computationally expensive. Recently, Lee et al. (2017) proposed a character level NMT that does address the long sequence problem by utilizing 1D CNNs to create different sized n-grams nodes, and compress the sequence with maxpool striding.

3. **Byte-Pair Encoding (BPE)** - Sennrich et al. (2016) proposed a way to represent language by breaking words down to subword units, encoding common pair of letters as a unique unit. This helps reduce the sequence length as well as the total vocabulary size. 

4. **Thai Character Cluster (TCC)**  - Theeramunkong et al. (2000) suggested a technique called 'Thai character clustering' (TCC) that groups Thai characters based on the Thai writing system into clusters that cannot be further separated. This is possible because in Thai, there are vowel and tone marks that cannot stand alone. This is similar to BPE/wordpiece, but rule based rather than data-driven. The english equivalent would be to call 'qu' a character cluster, since 'u' always follow 'q'.

    > Although commonly referred to as the "Thai alphabet", the script is in fact not a true alphabet but an abugida, a writing system in which each consonant may invoke an inherent vowel sound. In the case of the Thai script this is an implied 'a' or 'o'. Consonants are written horizontally from left to right, with vowels arranged above, below, to the left, or to the right of the corresponding consonant, or in a combination of positions. - Wikipedia



### Multi-lingual ###

Is meaning language bound, or do concepts exist in abstract which then get decoded to languages? For the machine learning field, the answer seems to be the latter, as performance can often be incrased by training on multiple source languages, and translating to one. 

The focus has mostly been on non-Asian languages though, especially those that share similar alphabets. I want to see whether SEA languages can learn from each other, especially those that do NOT share characters. Luckily, there are two SEA corpuses for TED Talk that are sizable enough, Thai and Vietnamese. 

Our experiment will train on the TH-EN  and VI-EN data and evaluate on TH-EN. We want to know whether the weights learned from another langauge can help translated our initial pair or not. At the word level, our vocabulary size will increase by  about twofold, since they do not share alphabets. 

We will also initialize every sentence with a token that indicates the source language, an idea pioneered by Google's multilingual [paper](https://arxiv.org/abs/1611.04558). Note though, that our RNN cells do not have skip connection like those in Google's paper. 



## Data

We will use TED Talk 2016's subtitle data set to train our data. Unlike the Thai data made available for the 2015 IWSLT evaluation campaign, our data does not come tokenized. To build up our corpus, the [WIT3]() script finds talks that exist in both languages, and finds parallel subtitle within the talk. Each 'sample segment' is a subtitle line. Sometimes it is a complete sentence and sometimes not. The script then reconstruct the segments into a sentence, based on ascii punctuations of the target language. This means you should not build sentence level parallel corpus, using the WIT3 script with languages like Thai or Chinese as the target language. Thai has no end of sentence markers, and Chinese does not use ascii punctuations. 

| Language Pair        |                       Sample segments | Total words with white space split() |
| -------------------- | ------------------------------------: | -----------------------------------: |
| Thai - English       | 187,731 segments <br>81,242 sentences |      324,981 (TH)<br> 1,383,482 (EN) |
| Thai - Vietnamese    |  151,814 segments<br>62,300 sentences |       257,909 (TH)<br>1,471,282 (VN) |
| Vietnamese - English | 271,848 segments<br>116,443 sentences |    2,006,934 (VN)<br> 2,629,575 (EN) |

Note that the word count for Thai using conventional split() is very low. This is because Thai does not use space, and needs to be further tokenized beyond the conventional split(). 

I mainly use segments as it reduces the sequence length of my data a lot and makes experimentation much more feasible (given time, memory constraint, and the fact that I own only 1 GPU). 



## Preprocessing

Each sentence is convereted to a vector with the maximum length of the longest sentence in the mini batch. I filter out samples with sequences that are too long. I set the acceptable maximum length of sequence for word-level at 30, subword units at 50, and characters at 250. 

**Note for BPE** - Sennrich's BPE script relies on being able to tokenize the data. For this, we preprocess our Thai data for BPE training with our word-level tokenizer. Without tokenizing the Thai text, the tokens that BPE ends up creating are characters. 

In contrast, TCC and character level models do not require any tokenization in preprocessing. 



## Evaluation ###

We use the [Moses](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) tokenizer and BLEU scipt to preprocess our result and the test set, before evaluating. 

I focus on translating with Thai as the source language, rather than the target, because the current script is built for languages that does not use spaces in an ambiguous way (ie: Thai) and Thai's tokenizer is among the things we are testing. 



## Model and hyper parameters ##

The models we will use are smaller versions of actual state-of-art NMTs. They are on average less than 1/4 the size of the original model. I base my model on two papers: 

1. [Luong et al.'s](http://aclweb.org/anthology/D15-1166) global attention model for word and subword level encoding, and decoding.
2. Lee et al.'s implementation of [CNN+GRU](https://arxiv.org/abs/1610.03017) for character level encoding. 

The differences are outlined in the tables below:

#### GRU Encoder for word, BPE, and TCC

|                | Luong's | Our model |
| -------------- | ------- | --------- |
| RNN Cell       | LSTM    | GRU       |
| Embedding size | 1000    | 256       |
| Hidden units   | 1000    | 256       |
| Layers         | 2       | 1         |
| Directions     | 1       | 1         |
| Dropout        | 0.2     | 0.2       |

#### CNN-GRU Encoder for characters

|                               | Lee's                                    | Our model                                |
| ----------------------------- | ---------------------------------------- | ---------------------------------------- |
| RNN Cell                      | GRU                                      | GRU                                      |
| Embedding size                | 128                                      | 128                                      |
| Hidden units                  | 512                                      | 256                                      |
| Layers                        | 1                                        | 1                                        |
| Direction                     | 2                                        | 1                                        |
| K-width <br>and # of filters  | [1,2,3,4,5,6,7,8] with<br />[200,200,250,250,300,300,350,350] | [1,2,3,4,5,6,7,8] with<br />[200,200,250,250,300,300,350,350] |
| Kernel stride <br>Pool stride | 1 <br>5                                  | 1<br>5                                   |
| Highway units                 | 4 layers of 2100 units(?*)               | 2100                                     |

*not stated in the paper how many hidden units were used. 

#### GRU Decoder with attention

|              | Luong's* | Lee's** | Our model |
| ------------ | -------- | ------- | --------- |
| RNN Cell     | LSTM     | GRU     | GRU       |
| Embedding    | 1000     | 512     | 256       |
| Hidden units | 1000     | 1024    | 256       |
| Layers       | 2        | 2       | 1         |
| Directions   | 1        | 1       | 1         |

\* Global attention is one of Luong's implementation of attention. In his paper, he showcases a variety them, with a more sophisticated version called "local attention" achieving the best NMT performance. 

\** Lee et al. actually uses [Bahdanau et al.](https://arxiv.org/abs/1409.0473) original version of attention, which is implemented slightly different and more complex. 

#### Optimizers

| Optimizing algorithm | Adam             |
| -------------------- | ---------------- |
| Learning rate        | 1e-4             |
| Batch size           | 128 - 256        |
| Gradient clipping    | 1                |
| Dropout              | 0.2 (inside GRU) |



## Result of experiments* ##

*std. devs to be included later with more runs

*BLEU scores are for validation set

### TH-EN

| Model      | NMT Baseline             | TCC2word                | BPE2word <br>(60k operations) | Char2word              |
| ---------- | :----------------------- | ----------------------- | ----------------------------- | ---------------------- |
| Input      | Word                     | TCC                     | BPE                           | Characters             |
| Vocab size | 28701 (TH)<br>36113 (EN) | 2737 (TH)<br>36113 (EN) | 19193 (TH)<br>36113 (EN)      | 195 (TH)<br>36113 (EN) |
| Encoder    | GRU                      | GRU                     | GRU                           | CNN-GRU                |
| Decoder    | GRU-attn                 | GRU-attn                | GRU-attn                      | GRU-attn               |
| BLEU Score | 10.7                     | 10.3                    | 9.88                          | 7.7                    |

#### Analysis

- **Score range**: For reference, our BLEU scores seem to be on the right track. The baseline BLEU score of 10.7 is about the same as this [paper's](https://arxiv.org/pdf/1606.07947.pdf) small model baseline for TH-EN, which used 2 layers attention LSTM with 100 hidden units and got 10.6 BLEU on the 2015 TED Talk corpus, which comes *pre segmented* by a Thai state research unit's state of art tokenizer (although this is not shared to other researchers). 
- **Subword units - TCC**: TCC achieves -0.4 BLEU while using only 1/10 the vocab size of the baseline word level model. This really shows really well how much information on the vocabulary side can actually be compressed, especially if given a good guideline. 
- **Subword units - BPE**: The data-driven BPE achieved about -0.4 to that of TCC though, despite using a larger set of vocabulary. I'm am actually somewhat surprised by the size of BPE vocabs, given the intial vocab size. The size was reduced to only 2/3. 

## TH-VN

| Model      | NMT Baseline             | TCC2word                | BPE2word<br>(60k operations) | Chars2word             |
| ---------- | ------------------------ | ----------------------- | ---------------------------- | ---------------------- |
| Input      | Word                     | TCC                     | BPE                          | Characters             |
| Vocab size | 25640 (TH)<br>17445 (VN) | 2653 (TH)<br>17445 (VN) | 17181 (TH)<br>17445 (VN)     | 190 (TH)<br>17455 (VN) |
| Encoder    | GRU                      | GRU                     | GRU                          | GRU+CNN                |
| Decoder    | GRU-attn                 | GRU-attn                | GRU-attn                     | GRU-attn               |
| BLEU       | 9.15                     | 8.62                    | 8.92                         | 7.31                   |

#### Analysis

- **Subword units - TCC/BPE**: It is interesting to note that for TH-VN, BPE outperforms TCC in terms of BLEU performance by +0.3 while being only -0.22 lower than the word level baseline. 

## VN-EN

| Model      | NMT Baseline             | BPE2word                  | Char2word              |
| ---------- | ------------------------ | ------------------------- | ---------------------- |
| Input      | Word                     | BPE                       | Characters             |
| Vocab size | 24964 (VN)<br>44264 (EN) | 32011 (VN) <br>44264 (EN) | 251 (VN)<br>44262 (EN) |
| Encoder    | GRU                      | GRU                       | CNN-GRU                |
| Decoder    | GRU-attn                 | GRU-attn                  | GRU                    |
| BLEU       | 18.7                     | 15.0                      | 15.1                   |

#### Analysis 

- **Baseline** : Wow, the initial BLEU score for VN-EN is really high! I suspect this is a combination of more samples (twice the amount of english words in VN-EN 2.6m compared TH-EN 1.3m), and the fact that you could tokenize VN with spaces. 
- **BPE vs Char** - Noticibly, BPE and char level performance are similar, for encoding vietnamese, unlike Thai. 

## Multilingual (with google style tokens)

| Model           | word2word                   | BPE2word                    | Char2word                 |
| --------------- | --------------------------- | --------------------------- | ------------------------- |
| Input<br>Output | TH + VN<br>EN               | TH + VN<br>EN               | TH + VN<br>EN             |
| Vocab size      | 54631 (TH+VN)<br>48145 (EN) | 51195 (TH+VN)<br>48415 (EN) | 327 (TH+VN)<br>48415 (EN) |
| Encoder         | GRU                         | GRU                         | CNN-GRU                   |
| Decoder         | GRU-attn                    | GRU-attn                    | GRU-attn                  |
| BLEU            | 9.27                        | 8.24                        | 8.23                      |

#### Analysis

- I am surprised to see that multilingual training did not help with the performance of TH-EN in our experiment, and this is true across all tokenizations. 
- Similar to VN-EN's experiment, BPE performs as well as character level model. 
- Though not shown here, but when training with multilingual dataset, the model takes a longer time to overfit than when trianing with bilingual pair. 

## Conclusion ##

**Thai, Tokenizers, small models** - Given relatively small models (256 x 2 layers with attention), word level tokenization seems to outperform other levels for Thai. One hypothesis is that the model has to spend less time figuring out spellings and can just dedicate itself entirely syntax and meanings matching.  

**Promises of subword-units** - The ability of TCC to reduce the vocabulary size by x10 while achieving slightly lower BLEU than word-level and comparable performance to BPE was very impressive to me. The only issue with TCC is that it is Thai language specific. 

BPE achieves similar results to TCC but requires the text to be preprocessed with a word tokenizer, so that a dictionary can be created. This means the score of BPE is somewhat dependent on the tokenizer's ability. 

**Character** - The character level models did not perform well in this experiment. One possible explanation is that character level NMTs require the model to be sufficiently big enough before they start to rival other NMTs. It also seems to  suffer when encoding Thai relative to BPE, unlike Vietnamese and multilingual. 

**Multilingual** 

The results of training our NMT on both TH and VN were not poor but did not see much gain over bilingual settings. 

Perhaps the model is too small for multilingual? Perhaps TH and VN don't share as much syntax as we hoped? 

Google was able to improve the score of JP-EN by training it with KO-EN though, so theoretically, differences in alphabets shouldn't make it impossible to improve the score. 



## Ideas for more experiments ##

**Tokernizers**

I think there is some room to create a new tokenizer that is data driven, unsupervised, not reliant on existing tokenizers, and not reliant on whitespace splits. 

It's also not entirely clear to me, why BPE / Wordpiece mostly limit itself to unigrams. The advantages that CNN models for text is their ability to create multiple n-grams features, and use them all jointly for analysis. 

Also, I wonder if the idea behind TCC, to identify unique character groups that cannot be further broken down', can be generalized to other non-abugida languages. In English, besides 'qu', are there other characters are are 'always' together and cannot be further broken down? Would it make sense to predefine a list of characters attached with vowels, like 'be' and try training over them? The downfall would be that we wouldn't be able to apply the same algorithm to every language. 

**Models**

-[ ] Try out transformer models. [link](https://arxiv.org/abs/1706.03762)


-[ ] See if characterlevel CNNs can be combined with transformers? Or whether it makes sense to? 

**Visualizing multilingual advantages and failures**

It would be really cool if we can visualize what training multilingual makes a model pay / not pay attention to that it had not / had before. I suppose a very simple way to do this is by mapping attention for a bilingual pair, and then compare it to a multilingual one. Another simple way is to print out test set sentences and see the difference in the result. But those seem more anecdotal, is it possible to visualize more broadly, at the language level? I think the closest might be the visualization in Google's [paper](https://arxiv.org/abs/1611.04558), using t-SNE projection to show syntax similarities, which is really cool but also somewhat hard to understand intuitively. 



## Quick References ##

1. A lot of ideas and design taken from [Lee's NYU code](https://github.com/nyu-dl/dl4mt-c2c). Particularly preprocessing. My implementation of his CNN-GRU model MIGHT have some mistakes though, since I don't code Theano and base my pytorch implementation mostly from reading his [paper](https://arxiv.org/abs/1610.03017).
2. TED Talk data from [WIT3](https://wit3.fbk.eu/). 
3. Python version of TCC implemented by [Korakot](https://github.com/korakot). 
4. Deepcut Thai tokenizer from [Rakpong](https://github.com/rkcosmos/deepcut)
5. Global attention from Luong [paper](http://aclweb.org/anthology/D15-1166). 
6. Moses for Tokernizer and BLEU script. [Link](https://github.com/moses-smt/mosesdecoder)
7. The original attention paper by [Bahdanau et al.](https://arxiv.org/abs/1409.0473) 
8. Academia, stackoverflow, google, and the internet for existing, and making it possible for a self-taught person like me to put something like this together.  

