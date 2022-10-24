# Text Augmentation using nlpaug python library

In some cases, we can’t find enough examples of specific information in our dataset to have enough examples to train a ML Model. We have to use Data Augmentation.

The goal with Data Augmentation is to increase the size of our dataset by fabricating new examples using the ones we have.

In this project, we are going to generate data from two text paragraphs each containing one example of label called Expansion Date.

We are going to use [nlpaug](https://github.com/makcedward/nlpaug) library.


**Sections** :

[Textual nlpaug Augmenters](#Textual-nlpaug-Augmenters)

- [Analyzing Augmenters](#Analyzing-Augmenters)

- [Ununsed Augmenters](#Ununsed-Augmenters)

- [Used Augmenters](#Used-Augmenters)

- [Used Approach](#Used-Approach)

- [Converting Expansion Date label](#Converting-Expansion-Date-label)

[Installation](#Installation)

[Discussion](#Discussion)

- [Data Augmentation limitations](#Data-Augmentation-limitations)

- [Other techniques](#Other-techniques)



## Textual nlpaug Augmenters

The [nlpaug](https://github.com/makcedward/nlpaug) library provides different augmenters for textual data by targeting characters, words or sentences.

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
    - **BackTranslationAug** : Augmenter that leverages two translation models for augmentation. For example, if the source is English, this augmenter translates source to German and then translates it back to English.
    - **ReservedAug** : Augmenter that applies target word replacement for augmentation. It can also be used to generate all possible combinations.

*   For sentences, the library provides 3 augmenters:
    - **ContextualWordEmbsForSentenceAug** : Augmenter that inserts sentence according to XLNet, GPT2 or DistilGPT2 prediction.
    - **AbstSummAug** : Augmenter that summarizes article by abstractive summarization method.
    - **LambadaAug** : Augmenter that uses language model to generate text and then uses classification model to retain high quality results.

## Analyzing Augmenters

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

- WordEmbsAug : We tested word embeddings to apply augmentation by inserting and substituting words, but it didn't give good results as the examples below:
  - word2vec : Example of an augmented phrase using GoogleNews-vectors-negative300.bin.gz model
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39. Sergia 1. bimbo The catcher Landlord hereby grants friedman to kl the Tenant the option Predator to take under lease
```
  - fasttext : Example of an augmented phrase using wiki-news-300d-1M.vec.zip model
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39. linguistically 1. The Landlord Lagalag hereby grants to the Tenant 18.82 the option triumphalistic to pre-1955 take under MikeDust lease
```
  - glove : Example of an augmented phrase using glove.6B.zip model
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : ahmes 39. negativity 1. The certeza Landlord hereby grants to pseudo-3d the Tenant the hatsopoulos option to take under toroc lease
```


- ReservedAug : It replaces reserved words. We can use SynonymAug and AntonymAug instead.


Regarding the sentence augmenters, we are not going to use LambadaAug and AbstSummAug.

- AbstSummAug : It summarizes articles, and we have short paragraphs.

- LambadaAug : It is a data augmentation method for text classification tasks as mentioned in [Do Not Have Enough Data? Deep Learning to the Rescue!](https://arxiv.org/pdf/1911.03118.pdf) paper. And in our case, it is not a classification task.

### Used Augmenters

In this section, we are going to present the augmenters that will be used in this project and an augmentation example of each one.


- SynonymAug : It substitutes words by WordNet's synonym like the example below.
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : thirty nine. ane. The Landlord hereby grants to the Renter the option to exact under lease
```

- AntonymAug : It substitutes words by antonym like the example below.
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39. 1. The Landlord hereby grants to the Tenant the option to disclaim under lease
```

- ContextualWordEmbsAug : It inserts words by using contextual word embeddings/ Language models. In the example below we used bert-base-uncased model (default model).
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : article 39. 1. the landlord hereby grants to the principal tenant lessee the option together to take part under a lease
```

- BackTranslationAug : In the following example, we translate the english sentence into german (using facebook/wmt19-en-de model), then we translate back the german sentence into english (using facebook/wmt19-de-en)
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the option to take under lease
Augmented : 39.1. The landlord hereby grants the tenant the opportunity to take over the lease
```

- ContextualWordEmbsForSentenceAug : It completes a sentence using language models. We tested xlnet-base-cased and gpt2 models. We are going to use the first one which returns acceptable results, unlike gpt2 which returns random words, as shown below.
  - Results of xlnet-base-cased model:
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the 
Augmented : 39.1. The Landlord hereby grants to the Tenant the title as written by or implied to her title hereunder to all the more valid titles on its property below and mentioned to such less valid articles below,
```
  - Results of gpt2:
```bash
Original : 39.1. The Landlord hereby grants to the Tenant the 
Augmented : 39.1. The Landlord hereby grants to the Tenant the % - and to is ( of in : - from / .'
```
  
### Used Approach

We want to augmente 2 text paragraphs each containing a mentioned label called Expansion Date, and turn them into 200 examples.

For each paragrach, we are going to augmenete the its first part (sentences before the label). Then, we are going to take the label and transform it into another type of date. Next, we are going to take the remaining of the paragraph (its second part : sentences after the label), and apply the data augmentation. Finally, we are going to concatenate the three generated components (first part, date and second part).

To augmente the paragraph's first and second parts, we are going to use a pipeline of augmenters for each part.

- The pipeline of the first paragraph:
  - Generating 5 examples using synonymAug Augmenter.
  - Generating 5 examples using antonymAug Augmenter.
  - Generating 5 examples using contextualWordEmbsAug Augmenter by injecting a new word to random position according to contextual word embeddings calculation.
  - Generating 5 examples using contextualWordEmbsAug Augmenter by replacing a word according to contextual embeddings calculation.
  - Generating 5 examples using ContextualWordEmbsForSentenceAug and synonymAug Augmenters. Since we have a label in our paragraph, we are going to replace the first sentences by a generated text (completeing the first words using XLNet model), and paraphrase the last sentence containing the label using synonymAug.
  - After applying the five previous steps, we have now 25 generated examples.
  - Backtranslating ( English - German - English) the original sentence and the generated 25 examples using BackTranslationAug Augmenter. We have now 51 generated examples.
  - Backtranslating ( English - Russian - English) the original sentence and the generated 51 examples using BackTranslationAug Augmenter. We have now 103 generated examples.
  - Backtranslating ( English - Arabic - English) the original sentence and the generated 103 examples using BackTranslationAug Augmenter. We have now 207 generated examples.
  - We are going to delete duplicates.
  
- The pipeline of the second paragraph:
  - Generating 5 examples using synonymAug Augmenter.
  - Generating 5 examples using antonymAug Augmenter.
  - Generating 5 examples using contextualWordEmbsAug Augmenter by injecting a new word to random position according to contextual word embeddings calculation.
  - Generating 5 examples using contextualWordEmbsAug Augmenter by replacing a word according to contextual embeddings calculation.
  - Generating 5 examples using ContextualWordEmbsForSentenceAug and synonymAug Augmenters. Since we have a label in our paragraph, we are going to paraphrase the first sentence containing the label using synonymAug, and replace the next sentences by a generated text (completeing the first words using XLNet model), .
  - After applying the five previous steps, we have now 25 generated examples.
  - Backtranslating ( English - German - English) the original sentence and the generated 25 examples using BackTranslationAug Augmenter. We have now 51 generated examples.
  - Backtranslating ( English - Russian - English) the original sentence and the generated 51 examples using BackTranslationAug Augmenter. We have now 103 generated examples.
  - Backtranslating ( English - Arabic - English) the original sentence and the generated 103 examples using BackTranslationAug Augmenter. We have now 207 generated examples.
  - We are going to delete duplicates.

Next, we are going to concatenate the 3 generated parts (first part, date and second part), and if the number of new paragraphs are less than 200 (after deleting duplicates), we are going to create new ones by randomly choosig sentences from the generating data and concatenating them.


### Converting Expansion Date label

To transform the label into another type of date, we applied two steps.

- The first step is changing the format of the date by randomly choosing one pattern from 13 defined patterns. For example, the "December 1, 1999" date is transformed in one of the following formats:
```bash
01-December-1999
12-01-1999
12/01/1999
01-Dec-1999
1999-12-01
Dec 01 1999
Dec. 01, 1999
December. 01, 1999
01 December 1999
1st December 1999
the First of December, 1999
December 1st, 1999
December the First, 1999
```

- The next step is randomly changing the day and the month of the provided date as the example below.
```bash
January the Fifteenth, 1999
```

## Installation 

- Install the requirements
```bash
pip install -r "requirements.txt"
```

- Run the script
```bash
python data_augmentation.py 
```


## Discussion

- To use BackTranslationAug augmenter, we have to install sacremoses, or an error will occur.


- After generating 200 exapmles for a paragraph, we found out that there are many duplicates especially after translating. So, we added more examples by randomly combining generating sentences from the first part and the second part of the paragraph.

- To generate the 200 augmented data, it takes approximatly 3 hours. The backtranslation augmenters take a lot of time to generate sentences.

- We cannot test different models from Hugging Face due to the limited number of language models supported by the [nlpaug](https://github.com/makcedward/nlpaug) library. For example, the ContextualWordEmbsForSentenceAug only supports XLNet and GPT2 models.


## Discussion

In this section, we are discussing some of the problems that such a data augmentation technique can create, and suggesting other techniques.

### Data Augmentation limitations

In the natural language processing (NLP) field, it is hard to augment text due to the high complexity of language. On the other hand, generating an augmented image in the computer vision area is relatively easier by flipping, adding salt, etc..

- Random token Replacement/Insertion using augmenters like SynonymAug, AntonymAug or ContextualWordEmbsAug may be a locally acceptable augmentation method but possibly disrupt the meaning of the whole sentence or the next and/or previous sentences. 
Also, such methods may ensure the validity of the augmented data, but also lead to insufficient semantic diversity. 
 
- Working in specialized domains such as those with domain-specific vocabulary and jargon (e.g. Law) can present challenges. Many pretrained models and external knowledge like WordNet cannot be effectively used. And this is can be applied to subdomains as well. For example, if the data to augment is about lease agreement, the augmented data will have the same topic. Hence, the trained model will probably not be able to generalize to new type of contracts during inference. And that, because of the golden rule in data science which is **garbage in garbage out**.


- The distribution of augmented data could be too similar from the original sentence when using token replacement or backtranslation, or too different from the original sentence when using Text Generation language models like GPT-2. This may lead to greater overfitting or poor performance through training on examples not representative of the given domain.


Finally, we cannot deny the importance of Data Augmentation for NLP. Furthermore, it reduces the cost of collection of data and labelling data, and it prevents data scarcity. However, gathering real data is more efficient. Moreover, by generating training data, the model can learn how to reverse-engineer the script.


### Other techniques

- Using Sequence-to-Sequence models such as T5-like models to paraphrase sentences. For example, we can use [prithivida/parrot_paraphraser_on_T5](https://huggingface.co/prithivida/parrot_paraphraser_on_T5) model from Hugging Face.

- Fine-tuning auto-regressive models on our own Dataset containing legal documents and contracts to generate data in the same domain as the training data.

- Using TF-IDF by training a model from scratch using publicaly available datasets like [albertvillanova/legal_contracts](https://huggingface.co/datasets/albertvillanova/legal_contracts). In this case, we can use TfIdfAug Augmenter from [nlpaug](https://github.com/makcedward/nlpaug) library.

- Testing other open source Frameworks/Libraries:

  - [SentAugment](https://github.com/facebookresearch/SentAugment) Data augmentation by retrieving similar sentences from larger datasets.
  - [faker](https://github.com/joke2k/faker) - Python package that generates fake data for you.
  - [textflint](https://github.com/textflint/textflint) - Unified Multilingual Robustness Evaluation Toolkit for NLP.
  - [Parrot](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser) - Practical and feature-rich paraphrasing framework.
  - [AugLy](https://github.com/facebookresearch/AugLy) - data augmentations library for audio, image, text, and video.
  - [TextAugment](https://github.com/dsfsi/textaugment) - Python 3 library for augmenting text for natural language processing applications.


- Generating adversarial examples using [TextAttack](https://github.com/dsfsi/textaugment).

- After generating new examples, we can check the similarity between the original sentence and the generating ones by adding an additional layer in our approach using for example [SentenceTransformers](https://www.sbert.net/) to choose the sentences with the highest similar meaning.

 
