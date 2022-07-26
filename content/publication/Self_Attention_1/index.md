---
abstract: 

# author_notes:
# - Equal contribution
# - Equal contribution
authors:
- admin
date: "2020-07-01T00:00:00Z"
doi: ""
featured: true
image:
  caption: 'Image credit: [****]("featured.jpeg")'
  focal_point: ""
  preview_only: true
projects:
# - example
publication: 
publication_short: In [*Towards Data Science*](https://towardsdatascience.com/)
publication_types:
# - "1"
# publishDate: "2017-01-01T00:00:00Z"
slides: 
summary: Breaking down the first part of transformer language models - the self-attention mechanism.
tags: ["NLProc", "Deep Learning"]
title: "1 | Basics of Self-Attention"
subtitle: ""
url_code: ""
url_dataset: ""
url_pdf: ""
url_poster: ""
url_project: ""
url_slides: ""
url_source: ""
url_video: ""
---

TL;DR — Transformers are an exciting and (**relatively**) new part of Machine Learning (ML) but there are a **lot** of concepts that need to be broken down before you can understand them. This is the first post in a column I’m writing about them. Here we focus on how the basic self-attention mechanism works, which is the first layer of a Transformer model. Essentially for each input vector Self-Attention produces a vector that is the weighted sum over the vectors in its neighbourhood. The weights are determined by the relationship or _connectedness_ between the words. This column is aimed at ML novices and enthusiasts who are curious about what goes on under the hood of Transformers.

Contents:
=========

1.  [Introduction](#cce2)
2.  [Self-Attention — the math](#2beb)
3.  [References](#c2e8)

1\. Introduction
================

Transformers are an ML architecture that have been used successfully in a wide variety of NLP tasks, especially sequence to sequence (seq2seq) ones such as machine translation and text generation. In seq2seq tasks, the goal is to take a set of inputs (e.g. words in English) and produce a desirable set of outputs (- the same words in German). Since their inception in 2017, they’ve usurped the dominant architecture of their day ([LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory)) for seq2seq and have become almost ubiquitous in any news about NLP breakthroughs (for instance OpenAI’s [GPT-2 even appeared in mainstream](https://www.vox.com/2019/5/15/18623134/openai-language-ai-gpt2-poetry-try-it) media!).

![](https://miro.medium.com/max/984/1*pblofc3psQrBkvXI4Jfxog.png)Fig 1.1 — machine translation (EN → DE)⁴

This column is intended as a very gentle, gradual introduction to the math, code and concept behind Transformer architecture. There’s no better place to start with than the attention mechanism because:

> The most basic transformers rely purely on attention **mechanisms³.**

2\. Self-Attention — the math
=============================

We want an ML system to learn the important relationships between words, similar to the way a human being understands words in a sentence. In Fig 2.1 you and I both know that “The” is referring to “animal” and thus should have a strong connection with that word. As the diagram’s colour coding shows, this system knows that there is some connection between “animal”, “cross”,“street” and “the” because they’re all _related_ to “animal”, the subject of the sentence. This is achieved through _Self-Attention.⁴_

![](https://miro.medium.com/max/1400/1*9XxSNAGInd3rbwTE_AwrQA.png)Fig 2.1 — which words does “The” pay **_attention_** to?⁴

At its most basic level, Self-Attention is a process by which one sequence of vectors _x_ is **encoded** into another sequence of vectors _z_ (Fig 2.2). Each of the original vectors is just a **block of numbers** that **represents a word.** Its corresponding _z_ vector represents both the original word _and_ its **relationship** with the other words around it.

![](https://miro.medium.com/max/1400/1*qeY6mWlzwkCIl2LhPN0zZQ.png)Fig 2.2: sequence of input vectors _x_ getting turned into another equally long sequence of vectors _z_

Vectors represent some sort of thing in a _space,_ like the flow of water particles in an ocean or the effect of gravity at any point around the Earth. You _can_ think of words as vectors in the total space of words. The direction of each word-vector _means_ something. Similarities and differences between the vectors correspond to similarities and differences between the words themselves (I’ve written about the subject before [here](https://medium.com/analytics-vidhya/ideas-for-using-word2vec-in-human-learning-tasks-1c5dabbeb72e)).

Let’s just start by looking at the first three vectors and only looking in particular at how the vector _x2_, our vector for “cat”, gets turned into _z2_. All of these steps will be repeated for _each_ of the input vectors.

First, we multiply the vector in our spotlight, _x2_, with all the vectors in a sequence, _including itself_. We’re going to do a product of each vector and the _transpose_ (the diagonally flipped version) of _x2_ (Fig 2.3). This is the same as doing a dot product and you can think of a dot product of two vectors as a measure of **how similar they are.**

![](https://miro.medium.com/max/1400/1*dVJGPnBgZAFy8MorveslUQ.png)Fig 2.3: transposed multiplication (superscript “T” = “transposed”)

The dot product of two vectors is proportional to the cosine of the angle between them (Fig 2.4) — so the more closely they align in direction, the larger the dot product. If they were pointing in the exact same direction then the angle A would be 0⁰ and a cosine of 0⁰ is equal to 1. If they were pointing in opposite directions (so that A = 180⁰) then the cosine would be -1.

![](https://miro.medium.com/max/1400/1*2c4vsG2yNRBQL8xsIYKuew.png)Fig 2.4 — dot product of two vectors

As an aside, note that the _operation_ we use to get this product between vectors is a hyperparameter we can choose. The dot product is just the simplest option we have and the one that’s used in [_Attention Is All You Need_](https://arxiv.org/pdf/1706.03762.pdf)_³_ (AIAYN)_._

If you want an additional intuitive perspective on this, [Bloem’s](http://www.peterbloem.nl/blog/transformers)¹ post discusses how self-attention is analogous to the way recommender systems determine the similarity of movies or users.

So we put one word under the spotlight at a time and determine its output from its neighbourhood of words. Here we’re only looking at the words before and after but we could choose to widen that window in the future.

![](https://miro.medium.com/max/900/1*RN9sHNRPhQu2atGXzTW5zg.png)Fig 2.5 — raw weights for each j-th vector

If the spotlit word is “cat”, the sequence of words we’re going over is “the”, “cat”, “sat”. We’re asking **how much attention the word “_cat”_ should pay to “_the”, “cat”_ and “_sat” respectively_** (similar to what we see in Fig 2.1).

Multiplying the transpose of our spotlit word vector and the sequence of words around it will give us a set of 3 _raw weights_ (Fig 2.5)_._ Each weight is proportional to how connected the two words are in meaning. We need to then normalise them so they are easier to use going ahead. We’ll do this using the [softmax formula (Fig 2.6).](https://en.wikipedia.org/wiki/Softmax_function) This converts a sequence of numbers to be within the range of 0, 1 where each output is proportional to the _exponential of the input number_. This makes our weights much easier to use and interpret.

![](https://miro.medium.com/max/1068/1*FM5PaDrHI31yoE8AwvMAWw.jpeg)Fig 2.6: normalising raw weights via softmax function

Now we take our normalised weights (one per every vector in the _j_ sequence), multiply them respectively with the _x_ input vectors, sum the products and bingo! We have an output _z_ vector, (Fig 2.5)! This is, of course, the output vector **just** for x2 (“cat”) — this operation will be repeated for every input vector in _x_ until we get the output sequence we saw in Fig 2.2.

![](https://miro.medium.com/max/1400/1*Q1d4gzdBleLgcMUrI58D8g.jpeg)Fig 2.7: Final operation to get our new sequence of vectors, _z_

This explanation so far may have raised some questions:

*   Aren’t the weights we calculated highly dependent on how we determined the original input vectors?
*   Why are we relying on the _similarity_ of the vectors? What if we want to find a connection between two ‘dissimilar’ words, such as the object and subject of “The cat sat on the matt”?

In the next post, we’ll address these questions. We’ll transform each vector for each of its different uses and thus define relationships between words _more precisely_ so that we can get an output more like Fig 2.8.

![](https://miro.medium.com/max/1400/1*al_9j5AzCoqPaTUMjFRkjQ.png)Fig 2.8 — which words is “cross” paying attention to in the **orange column** vs the **pink** one?

I hope you’ve enjoyed this post and I appreciate any amount of claps. Feel free to leave any feedback (positive or constructive) in the comments and I’ll aim to take it onboard as quickly as I can.

The people who helped my understanding the most and to whom I am very grateful are Peter Bloem (his [post](http://www.peterbloem.nl/blog/transformers) is a great start if, like me, you prefer a math-first approach to Machine Learning¹ ) and Jay Alammar (if you want a top-down view to start with, I recommend [his article](https://jalammar.github.io/illustrated-transformer/)²).

3\. References
==============

1.  Alammar J. _The Illustrated Transformer._ (2018)  [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) \[accessed 27th June 2020\]
2.  Bloem P. _Transformers from Scratch._ (2019) [http://www.peterbloem.nl/blog/transformers](http://www.peterbloem.nl/blog/transformers) .\[accessed 27th June 2020\]
3.  Vaswani A. et al. Dec 2017. _Attention is all you need_. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. [https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) \[accessed 27th June 2020\]. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
4.  Vaswani A. et al. Mar 2018 [arXiv:1803.07416](https://arxiv.org/abs/1803.07416) . [Interactive notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb?authuser=2#scrollTo=OJKU36QAfqOC): \[accessed 29th June 2020\]