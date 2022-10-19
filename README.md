# Project Description

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.


## Textual nlpaug Augmenters

The nlpaug library provides different augmenters for textual data by targeting characters, words or sentences.

*   For characters, the library provides 3 augmenters:
    - KeyboardAug : Augmenter that simulates typo error by random values. For example, people may type i as o incorrectly. One keyboard distance is leveraged to replace character by possible keyboard error.
    - OcrAug : Augmenter that simulates ocr error by random values. For example, OCR may recognize I as 1 incorrectly. Pre-defined OCR mapping is leveraged to replace character by possible OCR error.
    - RandomAug : Augmenter that generates character error by random values. For example, people may type i as o incorrectly.

*   For words, the library provides 10 augmenters:
- AntonymAug : Augmenter that substitutes opposite meaning word according to WordNet antonym
- ContextualWordEmbsAug : Augmenter that leverages contextual word embeddings to find top n similar word for augmentation. It uses language models (BERT, DistilBERT, RoBERTa or XLNet) to find out the most suitlabe word for augmentation
- RandomWordAug : Augmenter that applies randomly behavior for augmentation.
- SpellingAug : Augmenter that leverages pre-defined spelling mistake dictionary to simulate spelling mistake.
- SplitAug : Augmenter that applies word splitting for augmentation. It splits one word to two words randomly
- SynonymAug : Augmenter that substitutes similar word according to WordNet/ PPDB synonym.
- TfIdfAug : Augmenter that leverages TF-IDF statistics to insert or substitute word.
- WordEmbsAug : Augmenter that leverages word embeddings to find top n similar word for augmentation. It uses word embeddings (word2vec, GloVe or fasttext) to apply augmentation.
- BackTranslationAug : Augmenter that leverages two translation models for augmentation. For example, the source is English. This augmenter translate source to German and translating it back to English.
- ReservedAug : Augmenter that applies target word replacement for augmentation. It can also be used to generate all possible combinations.

*   For sentences, the library provides 3 augmenters:
- ContextualWordEmbsForSentenceAug : Augmenter that inserts sentence according to XLNet, GPT2 or DistilGPT2 prediction.
- AbstSummAug : Augmenter that summarizes article by abstractive summarization method.
- LambadaAug : Augmenter that uses language model to generate text and then uses classification model to retain high quality results.

## Choosing Augmenters

In our case, we want to apply Data Augmentation on paragraphs extracted from contracts and legal documents. Hence, we want to generate data that is understandable and closer to real data semantically using different words and sentences without lexical and syntaxic errors.


### Ununsed Augmenters

We are not going to focus on character augmenters because typo errors are generally used for chatting messages when people send messages quickly without rereading their phrases. 

Also, we are not going to use some of the word augmenters that add noise to training data instead of meaning including:

- SpellingAug : which substitutes word by spelling mistake words dictionary. In the example below, it replaces '1' by 'l', 'The' by 'Tge' and 'tht' and 'tem', 'to' by 'toa' and 'two'.. 
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39. l. Tge Landlord hereby grants toa tht Tenant tem option two taking ander lease
```

- RandomWordAug : which swap word randomly. In the example below, it replaces '.', 'to' and 'the' words randomly.
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : . 39 1 The. Landlord hereby to grants the the Tenant option to take under lease
```

- TfIdfAug : TfIdfAug has to be trained based on our data. But, we don't have a dataset beacuse we have only two paragraphs to be augmented in our training data.

- SplitAug : which splits one token to two tokens randomly. In the example below, it splits many words like 'Tenant' becomes 'Tena nt'.
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39. 1. The Landl ord hereby g rants to the Tena nt the option to t ake u nder le ase
```

- ReservedAug : It replaces reserved words. We can use SynonymAug and AntonymAug instead.


- LambadaAug : It is a data augmentation method for text classification tasks as mentioned in [Do Not Have Enough Data? Deep Learning to the Rescue!](https://arxiv.org/pdf/1911.03118.pdf) paper. And in our case, it is not a classification task.

### Used Augmenters


## Installation
The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install numpy requests nlpaug
```
or install the latest version (include BETA features) from github directly
```bash
pip install numpy git+https://github.com/makcedward/nlpaug.git
```
or install over conda
```bash
conda install -c makcedward nlpaug
```

If you use BackTranslationAug, ContextualWordEmbsAug, ContextualWordEmbsForSentenceAug and AbstSummAug, installing the following dependencies as well
```bash
pip install torch>=1.6.0 transformers>=4.11.3 sentencepiece
```

If you use LambadaAug, installing the following dependencies as well
```bash
pip install simpletransformers>=0.61.10
```

If you use AntonymAug, SynonymAug, installing the following dependencies as well
```bash
pip install nltk>=3.4.5
```

If you use WordEmbsAug (word2vec, glove or fasttext), downloading pre-trained model first and installing the following dependencies as well
```bash
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model

pip install gensim>=4.1.2
```

If you use SynonymAug (PPDB), downloading file from the following URI. You may not able to run the augmenter if you get PPDB file from other website
```bash
http://paraphrase.org/#/download
```

If you use PitchAug, SpeedAug and VtlpAug, installing the following dependencies as well
```bash
pip install librosa>=0.9.1 matplotlib
```

## Recent Changes

### 1.1.11 Jul 6, 2022
*   [Return list of output](https://github.com/makcedward/nlpaug/issues/302)
*   [Fix download util](https://github.com/makcedward/nlpaug/issues/301)
*   [Fix lambda label misalignment](https://github.com/makcedward/nlpaug/issues/295)
*   [Add language pack reference link for SynonymAug](https://github.com/makcedward/nlpaug/issues/289)


See [changelog](https://github.com/makcedward/nlpaug/blob/master/CHANGE.md) for more details.

## Extension Reading
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)
*   [How does Data Noising Help to Improve your NLP Model?](https://medium.com/towards-artificial-intelligence/how-does-data-noising-help-to-improve-your-nlp-model-480619f9fb10)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Unsupervied Data Augmentation](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
*   [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)

## Reference
This library uses data (e.g. capturing from internet), research (e.g. following augmenter idea), model (e.g. using pre-trained model) See [data source](https://github.com/makcedward/nlpaug/blob/master/SOURCE.md) for more details.

## Citation

```latex
@misc{ma2019nlpaug,
  title={NLP Augmentation},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpaug},
  year={2019}
}
```

This package is cited by many books, workshop and academic research papers (70+). Here are some of examples and you may visit [here](https://github.com/makcedward/nlpaug/blob/master/CITED.md) to get the full list.

### Workshops cited nlpaug
*   S. Vajjala. [NLP without a readymade labeled dataset](https://rpubs.com/vbsowmya/tmls2021) at [Toronto Machine Learning Summit, 2021](https://www.torontomachinelearning.com/). 2021

### Book cited nlpaug
*   S. Vajjala, B. Majumder, A. Gupta and H. Surana. [Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems](https://www.amazon.com/Practical-Natural-Language-Processing-Pragmatic/dp/1492054054). 2020
*   A. Bartoli and A. Fusiello. [Computer Vision–ECCV 2020 Workshops](https://books.google.com/books?hl=en&lr=lang_en&id=0rYREAAAQBAJ&oi=fnd&pg=PR7&dq=nlpaug&ots=88bPp5rhnY&sig=C2ue8Xxbu09l59nAMOcVxWYvvWM#v=onepage&q=nlpaug&f=false). 2020
*   L. Werra, L. Tunstall, and T. Wolf [Natural Language Processing with Transformers](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246/ref=sr_1_3?crid=2CWBPA8QG0TRU&keywords=Natural+Language+Processing+with+Transformers&qid=1645646312&sprefix=natural+language+processing+with+transformers%2Caps%2C111&sr=8-3). 2022

### Research paper cited nlpaug
*   Google: M. Raghu and  E. Schmidt. [A Survey of Deep Learning for Scientific Discovery](https://arxiv.org/pdf/2003.11755.pdf). 2020
*   Sirius XM: E. Jing, K. Schneck, D. Egan and S. A. Waterman. [Identifying Introductions in Podcast Episodes from Automatically Generated Transcripts](https://arxiv.org/pdf/2110.07096.pdf). 2021
*   Salesforce Research: B. Newman, P. K. Choubey and N. Rajani. [P-adapters: Robustly Extracting Factual Information from Language Modesl with Diverse Prompts](https://arxiv.org/pdf/2110.07280.pdf). 2021
*   Salesforce Research: L. Xue, M. Gao, Z. Chen, C. Xiong and R. Xu. [Robustness Evaluation of Transformer-based Form Field Extractors via Form Attacks](https://arxiv.org/pdf/2110.04413.pdf). 2021


## Contributions
<table>
  <tr>
    <td align="center"><a href="https://github.com/sakares"><img src="https://avatars.githubusercontent.com/u/1306031" width="100px;" alt=""/><br /><sub><b>sakares saengkaew</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/bdalal"><img src="https://avatars.githubusercontent.com/u/3478378?s=400&v=4" width="100px;" alt=""/><br /><sub><b>Binoy Dalal</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/emrecncelik"><img src="https://avatars.githubusercontent.com/u/20845117?v=4" width="100px;" alt=""/><br /><sub><b>Emrecan Çelik</b></sub></a><br /></td>
  </tr>
</table>ContextualWordEmbsForSentenceAug	
