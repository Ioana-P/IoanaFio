---
abstract: 
# TL;DR — [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA, sometimes LDirA/LDiA) is one of the most popular and interpretable generative models for finding **topics in text data**. I’ve provided an [example notebook](https://nbviewer.jupyter.org/github/Ioana-P/MLEng_vs_DScientist_analysis/blob/master/2_Topic_modelling.ipynb#topic=0&lambda=1&term=) based on web-scraped job description data. Although running LDA on a canonical dataset like [20Newsgroups](https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups) would’ve provided [clearer topics](https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/sklearn.ipynb) , it’s important to witness how difficult topic identification can be “in the wild”, and how you might not actually find clear topics — with unsupervised learning, you are _never guaranteed to find an answer!_
# author_notes:
# - Equal contribution
# - Equal contribution
authors:
- admin
date: "2020-09-26T00:00:00Z"
doi: ""
featured: false
image:
  caption: 'Image credit: [****](https://miro.medium.com/max/1400/1*Xs1Xe1Hh4P6IGyWN8fImXw.jpeg)'
  focal_point: ""
  preview_only: true
projects:
# - example
publication: 
publication_short: In [*Towards Data Science*](https://towardsdatascience.com/)
publication_types:
# - "1"
publishDate: "2020-09-26T00:00:00Z"
slides: 
summary: I explain the intuition behind LDirA and go through a worked example of applying and visualizing it to a canonical NLP dataset.
tags: ["NLProc", "Unsupervised Learning"]
title: "Latent Dirichlet Allocation: Intuition, math, implementation and visualisation with pyLDAvis"
url_code: ""
url_dataset: ""
url_pdf: ""
url_poster: ""
url_project: ""
url_slides: ""
url_source: ""
url_video: ""
---

TL;DR — [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA, sometimes LDirA/LDiA) is one of the most popular and interpretable generative models for finding **topics in text data**. I’ve provided an [example notebook](https://nbviewer.jupyter.org/github/Ioana-P/MLEng_vs_DScientist_analysis/blob/master/2_Topic_modelling.ipynb#topic=0&lambda=1&term=) based on web-scraped job description data. Although running LDA on a canonical dataset like [20Newsgroups](https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups) would’ve provided [clearer topics](https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/sklearn.ipynb) , it’s important to witness how difficult topic identification can be “in the wild”, and how you might not actually find clear topics — with unsupervised learning, you are _never guaranteed to find an answer!_

*   **Acknowledgement**: the greatest aid to _my_ understanding was Louis Serrano’s two videos on LDA (2020). A lot of the intuition section is based on his explanation, and I would urge you to visit his [video](https://www.youtube.com/watch?v=T05t-SqKArY) for a more thorough dissection.

![](https://miro.medium.com/max/1400/1*Xs1Xe1Hh4P6IGyWN8fImXw.jpeg)Fig 1.0 — the LDA “machine” producing documents

Contents:
=========

[Intuition](#f1d8)

[Maths](#7b8f)

[Implementation and visualisation](#b9d6)

Intuition
=========

Let’s say that you have a collection of different news articles (your _corpus_ of _documents_), and you suspect that there are several topics that come up frequently within said corpus — your goal is to find out what they are! To get there you make a few **key assumptions:**

*   _The d_[_istributional hypothesis_](https://en.wikipedia.org/wiki/Distributional_semantics#:~:text=The%20distributional%20hypothesis%20suggests%20that,occur%20in%20similar%20linguistic%20contexts.)_:_ Words that appear together frequently are likely to be close in meaning;
*   each topic is a mixture of different words (Fig 1.1);
*   each document is a mixture of different topics (Fig 1.2).

![](https://miro.medium.com/max/1400/1*bgPL1Ex8dfxBSM7bSE3HlA.jpeg)Fig 1.1 — Topics as a mixture of words

In Fig 1.1 you’ll notice that the topic “Health & Medicine” has various words associated with it to _varying degrees_ (“cancer” is more strongly associated than “vascular” or “exercise”). Note that different words can be associated with different topics, as with the word “cardio”.

![](https://miro.medium.com/max/1400/1*-dW-PbkYomLrP6XtNTYwHA.jpeg)Fig 1.2 — Document as a mixture of topics

In Fig 1.2 you’ll see that a single document can pertain to multiple topics (as colour-coded on the left). Words like “injury” and “recovery” might also belong to multiple topics (hence why I’ve coloured them in more than one colour).

Now LDA is a _generative model_ — it tries to determine the underlying mechanism that _generates_ the articles and the topics. Think of it as if there’s a machine with particular settings that spits out articles, but we can’t see the machine’s settings, only what it produces. LDA creates a set of machines with different settings and selects the one that gives the best-fitting results (Serrano, 2020). Once the best one is found, we take a look at its “settings” and we deduce the topics from that.

So what are these _settings_?

First, we have something called the _Dirichlet_ (pronounced like dee-reesh-lay) _prior_ of the topics. This is a number that says how _sparse_ or how _mixed_ up our topics are. In L Serrano’s video (which I highly recommend!) he illustrates how visually you can think of this as a triangle (Fig 1.3) where the dots represent the documents and their position with respect to the corners (i.e. the topics) represents the how they’re related to each of the topics (2020). So a dot that is very close to the “Sports” vertex will be almost entirely about sport.

![](https://miro.medium.com/max/1400/1*46pYVxXIOAL7qd40Bs_xHQ.jpeg)Fig 1.3 — Dirichlet distribution of topics

In the lefthand triangle the documents are fairly separated, most of them neatly tucked into their corners (this corresponds to a low Dirichlet prior, alpha<1); on the right they are in the middle and represent a more even mix of topics (a higher Dirichlet prior, alpha>1). Look at the document in Fig 1.2 and, given the mix of topics, have a think about where you think it would be placed in the triangle on the right (my answer is that it’d be the dot _just above_ the one closest to the Sports corner).

Second, we have the Dirichlet prior of the _terms_ (all the words in our vocabulary). This number (whose name is _beta)_ has almost exactly the same function as alpha — except that it determines how the **topics** are distributed amongst the **terms**_._

![](https://miro.medium.com/max/1400/1*ctgYvHaDDkcDKAzYcVigHg.jpeg)Fig 1.4 Dirichlet distribution of terms; the numbers are proportional to how much each word is associated with each respective topic

As we said before, the topics are assumed to be mixtures (more precisely, _distributions_) of different terms. In Fig 1.4 “Sports” is mostly drawn towards “injury”. “Health&Medicine” is torn between “cardio” and “injury” and has no association with the term “pray”.

_But wait, our vocabulary doesn’t consist of just 3 words!_ You’re right! We could have a vocabulary of _4 words_ (as shown in Fig 1.5)! Trouble is that visualising a typical vocabulary of _N_ words (where _N_ could be 10'000) would require a [generalised version of the triangle shape,](https://en.wikipedia.org/wiki/Simplex#The_standard_simplex) but in _N — 1_ dimensions (the term for this is an n-1 _simplex_). This is where the visuals stop and we trust that the maths of higher dimensions will function as expected. This also applies to the topics — very often we’ll find ourselves with more than 3 topics.

![](https://miro.medium.com/max/1400/1*iq3bjiBg_Pchh0upPmnmMQ.jpeg)Fig 1.5 — which topic is the red one, based on the distribution of terms?

An important clarification: in LDA we start with values of alpha and beta as hyperparameters, but these numbers _only_ tell us whether our dots (documents / topics) are **generally** concentrated in the middle of their triangles or closer to the corners. The _actual positions_ within the triangle (simplex) are guessed by the machine — the guesswork is not random, it’s heavily weighted by the Dirichlet priors.

So the machine creates the two Dirichlet distributions, _distributes_ the documents and topics on them and then _generates_ documents based on those distributions (Fig 1.6). So, how does the last step happen, the _generation_ part?

![](https://miro.medium.com/max/1400/1*Xs1Xe1Hh4P6IGyWN8fImXw.jpeg)Fig 1.6 — the LDA “machine” producing documents

Remember at the start we said that topics are seen as mixtures / distributions of words and documents as mixtures / distributions of topics? Going from left to right in Figure 1.7 we start with a document, somewhere in the triangle, torn between our 3 topics. If it’s near the “Sports” corner, this means that the document will be _mostly about Sports_, with some mentions of “Religion” and “Health&Medicine”. So we know the topic composition of the document → therefore we can estimate what _words_ will come up. We will be sampling (i.e. randomly pulling out) words mostly from Sports, some from Health&Medicine and a very small amount from Religion (Fig 1.7). Here’s a question for you: looking at the triangle at the bottom of Fig 1.7, do you think _word 2_ will come up or not?

![](https://miro.medium.com/max/1400/1*hDZIC8V8IyX-otJ1eblCuw.jpeg)Fig 1.7 — how the two Dirichlet distributions feed into our document generation

The answer is that **it might**: remember that topics are mixtures of words. You might be thinking that _word 2_ is very strongly related to the yellow (Religion) topic, and since this topic is very sparse in this document _word 2_ won’t come up as much. But remember that a. _word 2_ is also associated with the blue, Sports topic and b. the words are sample probabilistically, so every word has some non-zero chance of appearing.

The words in our final, generated document (on the right end of Fig 1.7) will be compared to the words in the original documents. We won’t get the same document, BUT when we compare a range of different LDA “machines” with a range of different distributions, we find that one of them was closer to generating the document than the others were and that’s the LDA model that we choose.

Maths
=====

A normal statistical language model assumes that you can generate a document by sampling from a probability distribution over words, i.e. for each word in our vocabulary there is an associated probability of that word appearing.

LDA adds a layer of complexity over this arrangement. It assumes a list of topics, _k_. Each document _m_ is a probability distribution over these _k_ topics, and each topic is a probability distribution over all the different terms in our vocabulary _V_. That is to say that each word has various probabilities of appearing in each topic.

The full probability formula that generates a document is in Figure 2.0 below. If we break this down, on the right hand side we have three product sums:

*   **Dirichlet distribution of topics over terms:** (corresponds to Fig 1.4 and 1.5) for each topic _i_ amongst _K topics_, what is the probability distribution of words for _i._
*   **Dirichlet distribution of documents over topics:** (corresponds to Fig 1.3) for each document _j_ in our corpus of size _M,_ what is the probability distribution of topics for _j._
*   **Probability of a topic appearing given a document X the probability of a word appearing given a topic:** (corresponding to the two rectangles in Fig 1.7) how likely is it that certain topics, _Z,_ appear in this document and then how likely is that certain words, _W,_ appear given those topics.

![](https://miro.medium.com/max/1400/1*pUTv6gS_8GDQodj4TlGTgw.jpeg)Fig 2.0 — LDA formula

The first two sums contain **symmetric** Dirichlet distributions which are prior probability distributions for our documents and our topics (Fig 2.1 shows a set of general Dirichlet distributions, including symmetric ones).

![](https://miro.medium.com/max/1280/1*YJbCG2oZI6prRgIBmHiQtg.png)Fig 2.1 — By Empetrisor — Own work, CC BY-SA 4.0, [https://commons.wikimedia.org/w/index.php?curid=49908662](https://commons.wikimedia.org/w/index.php?curid=49908662)

The 3rd sum contains two multinomial distributions, one over topics and one over words — i.e. we sample topics from a probability distribution of them and then for each topic instance we sample words from a probability distribution of words for that particular topic.

As was mentioned at the end of the Intuition section, using the final probability we try to generate the same distribution of words as the one that we get in our original documents. The probability of achieving this is _very, very low_, but for some values of alpha and beta the probability will be less low.

Interpreting an LDA model and its topics
----------------------------------------

What metrics do we use for finding our latent topics? As Shirley and Sievert note:

> “To interpret a topic, one typically examines a ranked list of the most probable terms in that topic, \[…\]. The problem with interpreting topics this way is that common terms in the corpus often appear near the top of such lists for multiple topics, making it hard to differentiate the meanings of these topics.” (2014)

That is exactly the problem we’ve stumbled into in the next section, _Implementation_. Therefore we use an alternative metric for interpreting our topics — _relevance_ (Shirley and Sievert, 2014).

Relevance
---------

This is an adjustable metric that balances a term’s frequency in a particular topic against the term’s frequency across the whole corpus of documents.

In other words, if we have a term that’s quite popular in a topic, relevance allows us to gauge how much of its popularity is due to it being very specific to that topic and how much of it is due to it just being a work that appears _everywhere._ An example of the latter would be “learning” in the job description data. When we adjust relevance with a lower lambda (i.e. penalising terms that just happen to be frequent across **all** topics), we see that “learning” is not that special a term, and it only comes up frequently because of its prevalence across the corpus.

The mathematical definition of relevance is:

![](https://miro.medium.com/max/1172/0*tL0f-BtwU3oSv-8-)

*   _r —_ relevance
*   _⍵ —_ a term in our vocabulary
*   _k —_ a topic amongst the ones our LDA has produced
*   _λ —_ the adjustable weight parameter
*   𝝓kw — probability of a term appearing in a particular topic
*   **_p_**_w —_ the probability of a term appearing inside the corpus as a whole

Apart from lambda, _λ,_ all the terms are derived from the LDA data and model. We adjust lambda in the next section to help us derive more useful insights. The original paper authors kept lambda in the range of 0.3 to 0.6 (Shirley and Sievert, 2014).

Implementation and visualisation
================================

The implementation of sklearn’s LatentDirichletAllocation model follows the pattern of most sklearn models. In my [notebook](https://nbviewer.jupyter.org/github/Ioana-P/MLEng_vs_DScientist_analysis/blob/master/2_Topic_modelling.ipynb#topic=0&lambda=1&term=), I:

1.  Pre-processed my text data,
2.  Vectorised it (resulting in a document-term matrix),
3.  Fit\_transformed it using LDA and then
4.  Inspected the results to see if there are any emergent, identifiable topics.

The last part is highly subjective (remember this is _unsupervised learning_) and is not guaranteed to reveal anything really interesting. Furthermore the ability to identify topics (like clusters) depends on your domain knowledge of the data. I recommend also altering the alpha and beta parameters to match your expectations of the text data.

The data I’m using is job post description data from indeed.co.uk. The dataframe has many other attributes than text, including whether I used the search terms “data scientist”, “data analyst” or “machine learning engineer”. Can we find some of the original search categories in our LDA topics?

In the gist below you’ll see that I’ve vectorised my data and passed it to an LDA model (this happens under the hood of the data\_to\_lda function).

Running this code and the print\_topics function will produce something like this:

```
Topics found via LDA on Count Vectorised data for ALL categories:  
  
Topic #1:  
software; experience; amazon; learning; opportunity; team; application; business; work; product; engineer; problem; development; technical; make; personal; process; skill; working; science  
  
Topic #2:  
learning; research; experience; science; team; role; work; working; model; skill; deep; please; language; python; nlp; quantitative; technique; candidate; algorithm; researcherTopic #3:  
learning; work; team; time; company; causalens; business; high; platform; exciting; award; day; development; approach; best; holiday; fund; mission; opportunity; problem  
  
Topic #4:  
client; business; team; work; people; opportunity; service; financial; role; value; investment; experience; firm; market; skill; management; make; global; working; support...
```

The “print\_topics” function gives the terms for each topic in decreasing order of probability, which **can** be informative. It’s at this stage that we can **start** trying to label the emergent, latent topics from our model. For instance, Topic 1 seems to be related mildly related to ML engineer skills and requirements (the mention of “amazon” relates to using AWS — this is something I found from the EDA stage of the project in another notebook); meanwhile, Topic 4 clearly has a more client-facing or business-oriented theme, given terms like “market”, “financial”, “global”.

Now those two categories might seem a bit far-fetched to you and that’s a fair criticism. You may also have noticed that using this method for topic determination is hard. So, let’s turn to pyLDAvis!

pyLDAvis
--------

Using pyLDAvis, the LDA data (which in our case, was 10-dimensional) has been decomposed via PCA (principal component analysis) to be only 2-dimensional. Thus it has been flattened for the purposes of visualisation. We have ten circles and the center of each circle represents the position of our topic in the latent feature space; the distances between topics illustrates how (dis)similar the topics are and the area of the circles is proportional to how many documents feature each topic.

Below I’ve shown how you insert an already trained sklearn LDA model in pyLDAvis. Thankfully the [people responsible for adapting the original LDAvis](https://github.com/bmabey/pyLDAvis) (which was R model) to python made it communicate efficiently with sklearn.

And in Fig 3.0 is the plot we generate:

![](https://miro.medium.com/max/1400/1*e9Fj031z3H1s_eNx_KnfWg.png)Fig 3.0 — pyLDAvis interactive plot

**Interpreting pyLDAvis plots**

The LDAvis plot comes in two parts — a 2-dimensional ‘flattened’ replotting of our n-dimensional LDA data and an interactive, varying horizontal bar-chart of term distributions. Both of these are shown in Fig A1.0. One important feature to note is that the right-hand bar chart shows the terms in a topic in _decreasing order of relevance_, but the bars indicate the frequency of the terms. The red section represents the term frequency purely within the particular topic; the red and blue represent the overall term frequency within the corpus of documents.

**Adjusting** _λ (lambda)_

If we set λ equal to 1, then our relevance is given purely by the probability of the word to that topic. Setting it to 0 will result in our relevance being dictated by specificity of that word to the topic — this is because the right hand term divides the probability of a term appearing in a particular topic divided by the probability of the word appearing generally — thus, highly frequent words (such as ‘team’, ‘skill’, ‘business’) will be downgraded heavily in relevance when we have a lower _λ_ value.

![](https://miro.medium.com/max/1400/1*gZJYETiTTlLPyi2VsXam9Q.png)Fig 3.1 — setting lambda to 1

In Fig 3.1 _λ_ was set to 1 and you can see that the terms tend to match the ones that dominate across the board generally (i.e. like in our print-outs of the most popular terms for each topic). This was only done for topic 1, but when I changed topic the distribution of top-30 most relevant terms barely changed at all!

Now, in Fig 3.2 _λ_ was set to 0 and the terms changed completely!

![](https://miro.medium.com/max/1400/1*vHiv2kJNqRAZdsC23_O3sg.png)Fig 3.2 — lambda set to 0

Now we have highly specific terms, but pay attention to the scale at the top — the most relevant word appears about 60 times. That’s quite a come down after over 6000! Also, these words won’t necessarily tell us anything interesting. If you select a different topic with this lambda value you will keep getting junk terms that aren’t necessarily that important.

In Fig 3.3 I’ve set lambda to 0.6 and I am exploring topic 2. Right off the bat there is a significant theme here surrounding engineer work, with terms like “aws”, “cloud” and “platform”.

![](https://miro.medium.com/max/1400/1*6Z1RjZ39WG4OCe6uyFE4uA.png)Fig 3.3 — lambda = 0.6

Another great thing that you can do with pyLDAvis is visually inspect the conditional topic distribution given a word, simply by hovering over the word (Fig 3.4). Below we can see just how much “NLP” is split amongst several topics — not a lot! This gives me further reason to believe that topic 6 is focused on NLP and text-based work (terms like “speech”, “language”, “text” also help in that regard). An interesting insight for me is the fact that “research” and “PhD” co-occur so strongly in this topic.

![](https://miro.medium.com/max/1400/1*lsD8XKNR7YSUlqkcVp-ZjA.png)Fig 3.4 — conditional topic distribution for “NLP”

Does this mean that NLP-focussed roles in the industry demand higher education than other roles? Do they demand previous research experience more often than other roles? Are NLP roles perhaps more fixated on experimental techniques and thus require someone with knowledge of the cutting edge?

While the interactive plot generated cannot deliver concrete answers, what it can do is provide us with a starting position for further investigation. If you’re in an organisation where you can run topic modelling, you can use LDA’s latent themes to inform survey-design, A/B testing or even correlate it with other available data to find interesting correlations!

I wish you the best of luck in topic modelling. If you’ve enjoyed this lengthy read, please give me as many claps as you think are appropriate. If you have knowledge of LDA and think I’ve gotten something **even partially wrong** please leave me a comment (feedback is a gift and all that)!

**References**
--------------

1.  Serrano L. (2020). Accessed online: [Latent Dirichlet Allocation (Part 1 of 2)](https://www.youtube.com/watch?v=T05t-SqKArY)
2.  Sievert C. and Shirley K (2014). _LDAvis: A method for visualizing and interpreting topics._ Accessed online: [Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)

