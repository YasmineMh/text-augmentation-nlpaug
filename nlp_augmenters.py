import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


# Implement a class to customize used methods of nlpaug library
class NLPAugmenters:
    def __init__(self) -> None:
        pass

    def synonymAugmenter(
        self, text: str, aug_p: float = 0.5, number_generated_examples: int = 5
    ):
        """
        Augmenter that substitutes similar word according to WordNet synonym

        Args:
            text: Sentence to be augmented
            aug_p: Percentage of word will be augmented
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        synonymAug = naw.SynonymAug(aug_src="wordnet", aug_p=aug_p)
        return synonymAug.augment(text, n=number_generated_examples)

    def antonymAugmenter(self, text: str, number_generated_examples: int = 5):
        """
        Augmenter that substitutes opposite meaning word according to WordNet antonym

        Args:
            text: Sentence to be augmented
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        antonymAug = naw.AntonymAug()
        return antonymAug.augment(text, n=number_generated_examples)

    def contextualWordEmbsAugmenter(
        self,
        text: str,
        model_path: str = "nlpaueb/legal-bert-base-uncased",
        action: str = "insert",
        number_generated_examples: int = 5,
    ):
        """
        Augmenter that leverages contextual word embeddings to find top n similar word for augmentation

        Args:
            text: Sentence to be augmented
            model_path: Model name or model path. It used transformers to load the model. The default model we use is nlpaueb/legal-bert-base-uncased, which has been pre-trained on 12 GB of diverse English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources
            action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced according to contextual embeddings calculation
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        contextualWEAug = naw.ContextualWordEmbsAug(
            model_path=model_path, action=action
        )
        generated_data = contextualWEAug.augment(text, n=number_generated_examples)
        cleaned_data = [sent.replace("[UNK]", "") for sent in generated_data]
        return cleaned_data

    def BackTranslationAugmenter(
        self,
        text: str,
        from_model_name: str = "facebook/wmt19-en-de",
        to_model_name: str = "facebook/wmt19-de-en",
    ):
        """
        Augmenter that leverages two translation models for augmentation

        Args:
            text: Sentence to be augmented
            from_model_name: Model name or model path. It used transformers to load the model. The default model we use is nlpaueb/legal-bert-base-uncased, which has been pre-trained on 12 GB of diverse English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources
            to_model_name: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced according to contextual embeddings calculation

        Returns:
            List of augmented sentences

        """
        backTranslationAug = naw.BackTranslationAug(
            from_model_name=from_model_name, to_model_name=to_model_name
        )
        return backTranslationAug.augment(text)

    def ContextualWordEmbsForSentenceAugmenter(
        self,
        text: str,
        model_path: str = "xlnet-base-cased",
        number_generated_examples: int = 5,
    ):
        """
        Augmenter that inserts sentence according to XLNet prediction

        Args:
            text: Sentence to be augmented
            model_path: Model name or model path. It used transformers to load the model. The default model we use is nlpaueb/legal-bert-base-uncased, which has been pre-trained on 12 GB of diverse English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources
            action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced according to contextual embeddings calculation
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        contextualWEFSAug = nas.ContextualWordEmbsForSentenceAug(model_path=model_path)
        return contextualWEFSAug.augment(text, n=number_generated_examples)

    def generate_text_before_label_with_autoregressive_model(
        self, text: str, text_limit: int = 11, number_generated_examples: int = 5
    ):
        """
        Augmenter that inserts sentence according to XLNet.
        It completes the first sentence containing text_limit words.
        And for the last sentence which contains the date label, the augmenter uses synonymAugmenter to paraphrase it

        Args:
            text: Sentence to be augmented
            text_limit: number of words of the sentence to be completed by the auto-regressive model
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        # Extract the first words of the first sentence and use them as input for the XLNet model
        tokens = text.split()
        if len(tokens) >= text_limit:
            first_sentence_to_be_generated = " ".join(tokens[:text_limit]) + " "
        else:
            first_sentence_to_be_generated = " ".join(tokens[: len(tokens)]) + " "
        # Complete the sentence
        generated_sentences = self.ContextualWordEmbsForSentenceAugmenter(
            first_sentence_to_be_generated,
            number_generated_examples=number_generated_examples,
        )
        # Extract the last sentence that will be paraphrased
        index_last_sentence = text.rfind(".")
        # if there is only one sentence in text, we just complete the sentence
        if index_last_sentence == -1:
            concat_lists = generated_sentences
        else:
            last_part = text[index_last_sentence + 2 :]
            completed_sentences = self.synonymAugmenter(
                last_part,
                aug_p=0.6,
                number_generated_examples=number_generated_examples,
            )
            concat_lists = []
            for index, i in enumerate(generated_sentences):
                new_sentence = i
                if i[-1] != ".":
                    new_sentence += "."
                new_sentence += " " + completed_sentences[index]
                concat_lists.append(new_sentence)
        return concat_lists

    def generate_text_after_label_with_autoregressive_model(
        self, text: str, text_limit: int = 9, number_generated_examples: int = 5
    ):
        """
        Augmenter that inserts sentence according to XLNet.
        It completes the second sentence containing text_limit words.
        And for the first sentence which contains the date label, the augmenter uses synonymAugmenter to paraphrase it

        Args:
            text: Sentence to be augmented
            text_limit: number of words of the sentence to be completed by the auto-regressive model
            number_generated_examples: Number of sentences to be augmented

        Returns:
            List of augmented sentences

        """
        index_after_first_sentence = text.find(".")
        # if there is only one sentence in text, we just complete the sentence
        if index_after_first_sentence == -1:
            tokens = text.split()
            if len(tokens) >= text_limit:
                sentence_to_be_generated = " ".join(tokens[:text_limit]) + " "
            else:
                sentence_to_be_generated = " ".join(tokens[: len(tokens)]) + " "
            first_sentence_to_be_generated = []
        else:
            # Paraphrase the first sentence
            first_part = text[0:index_after_first_sentence]
            first_sentence_to_be_generated = self.synonymAugmenter(
                first_part,
                aug_p=0.6,
                number_generated_examples=number_generated_examples,
            )
            # Extract the second sentence
            tokens = text[index_after_first_sentence + 2 :].split()
            if len(tokens) >= text_limit:
                sentence_to_be_generated = " ".join(tokens[:text_limit]) + " "
            else:
                sentence_to_be_generated = " ".join(tokens[: len(tokens)]) + " "
        # Complete the sentence
        generated_sentences = self.ContextualWordEmbsForSentenceAugmenter(
            sentence_to_be_generated,
            number_generated_examples=number_generated_examples,
        )
        # Concatenate the paraphrased sentence and the compeleted sentence
        if first_sentence_to_be_generated:
            concat_lists = [
                i + ". " + generated_sentences[index]
                for index, i in enumerate(first_sentence_to_be_generated)
            ]
        else:
            concat_lists = generated_sentences
        return concat_lists

    def augment_data_using_de_backtranslation(
        self, original_sentence: str, data_to_augment: list
    ):
        """
        Augmenter that leverages two translation models for augmentation.
        This augmenter translates English to German, then translates German back to English.

        Args:
            original_sentence: the original sentence used to create new examples by other augmenters
            data_to_augment: List of augmented data from original_sentence

        Returns:
            List of augmented sentences

        """
        de_aug_data = []
        de_aug_data.append(
            self.BackTranslationAugmenter(
                original_sentence, "facebook/wmt19-en-de", "facebook/wmt19-de-en"
            )[0]
        )
        for i in data_to_augment:
            de_aug_data.append(
                self.BackTranslationAugmenter(
                    i, "facebook/wmt19-en-de", "facebook/wmt19-de-en"
                )[0]
            )
        return list(set(de_aug_data))

    def augment_data_using_ru_backtranslation(
        self, original_sentence: str, data_to_augment: list
    ):
        """
        Augmenter that leverages two translation models for augmentation.
        This augmenter translates English to Russian, then translates Russian back to English.

        Args:
            original_sentence: the original sentence used to create new examples by other augmenters
            data_to_augment: List of augmented data from original_sentence

        Returns:
            List of augmented sentences

        """
        ru_aug_data = []
        ru_aug_data.append(
            self.BackTranslationAugmenter(
                original_sentence, "facebook/wmt19-en-ru", "facebook/wmt19-ru-en"
            )[0]
        )
        for i in data_to_augment:
            ru_aug_data.append(
                self.BackTranslationAugmenter(
                    i, "facebook/wmt19-en-ru", "facebook/wmt19-ru-en"
                )[0]
            )
        return list(set(ru_aug_data))

    def augment_data_using_ar_backtranslation(
        self, original_sentence: str, data_to_augment: list
    ):
        """
        Augmenter that leverages two translation models for augmentation.
        This augmenter translates Arabic to German, then translates Arabic back to English.

        Args:
            original_sentence: the original sentence used to create new examples by other augmenters
            data_to_augment: List of augmented data from original_sentence

        Returns:
            List of augmented sentences

        """
        ar_aug_data = []
        ar_aug_data.append(
            self.BackTranslationAugmenter(
                original_sentence,
                "Helsinki-NLP/opus-mt-en-ar",
                "Helsinki-NLP/opus-mt-ar-en",
            )[0]
        )
        for i in data_to_augment:
            ar_aug_data.append(
                self.BackTranslationAugmenter(
                    i, "Helsinki-NLP/opus-mt-en-ar", "Helsinki-NLP/opus-mt-ar-en"
                )[0]
            )
        return list(set(ar_aug_data))
