# Text Augmentation using nlpaug python library

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
  


### Converting Expansion Date label


## Installation


## Issues we run into

## Discussion

In this section, we are discussing some of the problems that such a data augmentation technique can create, and suggesting other techniques.

### Data Augmentation limitations

In the natural language processing (NLP) field, it is hard to augment text due to the high complexity of language. On the other hand, generating an augmented image in the computer vision area is relatively easier by flipping, adding salt, etc..

Random token Replacement/Insertion using augmenters like SynonymAug, AntonymAug or ContextualWordEmbsAug may be a locally acceptable augmentation method but possibly disrupt the meaning of the whole sentence or the next and/or previous sentences. 
Also, such methods may ensure the validity of the augmented data, but also lead to insufficient semantic diversity. 
 
Working in specialized domains such as those with domain-specific vocabulary and jargon (e.g. Law) can present challenges. Many pretrained models and external knowledge like WordNe cannot be effectively used. And this is can be applied to sub-domains as well. For example, if the data to augment is about lease agreement, the augmented data will have the same topic. Hence, the trained model will probably not be able to generalize to new type of contracts during inference. And that, because of the golden rule in data science which is **garbage in garbage out**.


The distribution of augmented data could be too similar from the original sentence when using token replacement or back-translation or too different from the original sentence when using Text Generation language models like GPT-2. This may lead to greater overfitting or poor performance through training on examples not representative of the given domain.


Finally, we cannot deny the importance of Data Augmentation for NLP. Furthermore, it reduces the cost of collection of data and labelling data, and it prevents data scarcity. However, gathering real data is more efficient. Moreover, by generating training data, the model can learn how to reverse-engineer the script.


### Other techniques

- Using Sequence-to-Sequence models such as T5-like models to paraphrase sentences. For example, we can use [prithivida/parrot_paraphraser_on_T5](https://huggingface.co/prithivida/parrot_paraphraser_on_T5) model from Hugging Face.

- Fine-tuning auto-regressive models on our own Dataset containing legal documents and contracts to generate data in the same domain as the training data.

- Using TF-IDF by training a model from scratch using publicaly available datasets like [albertvillanova/legal_contracts](https://huggingface.co/datasets/albertvillanova/legal_contracts). In this case, we can use TfIdfAug Augmenter from nlpaug library.

- Testing other open source Frameworks/Libraries:

  - [SentAugment](https://github.com/facebookresearch/SentAugment) Data augmentation by retrieving similar sentences from larger datasets.
  - [faker](https://github.com/joke2k/faker) - Python package that generates fake data for you.
  - [textflint](https://github.com/textflint/textflint) - Unified Multilingual Robustness Evaluation Toolkit for NLP.
  - [Parrot](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser) - Practical and feature-rich paraphrasing framework.
  - [AugLy](https://github.com/facebookresearch/AugLy) - data augmentations library for audio, image, text, and video.
  - [TextAugment](https://github.com/dsfsi/textaugment) - Python 3 library for augmenting text for natural language processing applications.


- Generating adversarial examples using [TextAttack](https://github.com/dsfsi/textaugment).

- After generating new examples, we can check the similarity between the original sentence and the generating ones by adding an additional layer in our approch using for example [SentenceTransformers](https://www.sbert.net/) to choose the sentences with the highest similar meaning.

 
