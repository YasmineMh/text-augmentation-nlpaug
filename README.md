# Project Description

Description


## Textual nlpaug Augmenters

The nlpaug library provides different augmenters for textual data by targeting characters, words or sentences.

*   For characters, the library provides 3 augmenters:
    - **KeyboardAug** : Augmenter that simulates typo error by random values. For example, people may type i as o incorrectly. One keyboard distance is leveraged to replace character by possible keyboard error.
    - **OcrAug** : Augmenter that simulates ocr error by random values. For example, OCR may recognize I as 1 incorrectly. Pre-defined OCR mapping is leveraged to replace character by possible OCR error.
    - **RandomAug** : Augmenter that generates character error by random values. For example, people may type i as o incorrectly.

*   For words, the library provides 10 augmenters:
    - **AntonymAug** : Augmenter that substitutes opposite meaning word according to WordNet antonym
    - **ContextualWordEmbsAug** : Augmenter that leverages contextual word embeddings to find top n similar word for augmentation. It uses language models (BERT, DistilBERT, RoBERTa or XLNet) to find out the most suitlabe word for augmentation
    - **RandomWordAug** : Augmenter that applies randomly behavior for augmentation.
    - **SpellingAug** : Augmenter that leverages pre-defined spelling mistake dictionary to simulate spelling mistake.
    - **SplitAug** : Augmenter that applies word splitting for augmentation. It splits one word to two words randomly
    - **SynonymAug** : Augmenter that substitutes similar word according to WordNet/ PPDB synonym.
    - **TfIdfAug** : Augmenter that leverages TF-IDF statistics to insert or substitute word.
    - **WordEmbsAug** : Augmenter that leverages word embeddings to find top n similar word for augmentation. It uses word embeddings (word2vec, GloVe or fasttext) to apply augmentation.
    - **BackTranslationAug** : Augmenter that leverages two translation models for augmentation. For example, the source is English. This augmenter translate source to German and translating it back to English.
    - **ReservedAug** : Augmenter that applies target word replacement for augmentation. It can also be used to generate all possible combinations.

*   For sentences, the library provides 3 augmenters:
    - **ContextualWordEmbsForSentenceAug** : Augmenter that inserts sentence according to XLNet, GPT2 or DistilGPT2 prediction.
    - **AbstSummAug** : Augmenter that summarizes article by abstractive summarization method.
    - **LambadaAug** : Augmenter that uses language model to generate text and then uses classification model to retain high quality results.

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


