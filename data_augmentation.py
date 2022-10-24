from nlp_augmenters import *
from transform_date_format import *

import random
import json


dataset = [
    {
        "paragraph": '39.1. The Landlord hereby grants to the Tenant the option to take under lease, subject and subordinate to the Qualified Encumbrances, the space substantially as shown cross-hatched on the diagram attached hereto as Exhibit A-3 and designated as \'D\' on the roof of the Building (herein called the "Additional Penthouse Space") subject to all the same terms and conditions of this Lease applicable to the Penthouse Space. The term with respect to the Additional Penthouse Space shall commence, and the Additional Penthouse Space shall be added to the Penthouse Space, on the date which is the earlier to occur of (a) December 1, 1999 and (b) the date upon which the Tenant first occupies the Premises for the conduct of its business, subject to Article Two of this Lease (the "Additional Penthouse Space Term Commencement Date"). The Tenant may exercise the option granted pursuant to this Section 39.1 (if at all) only by notifying the Landlord, in writing, not later than October 1, 1999.',
        "date_index_start": 615,
        "date_index_end": 631,
    },
    {
        "paragraph": 'Tenant shall expand the size of the Premises to include the balance of the space on the sixth (6/th/) floor of the Building as of the date upon which such space is delivered to Tenant by Landlord (the "Expansion Space Commencement Date") pursuant to the terms hereof. The parties estimate that the Expansion Space Commencement Date shall be September 6, 2001 (the "Target Date"). In connection therewith, it is understood and agreed that Landlord may deliver the Expansion Space to Tenant as early as three (3) months prior to the Target Date or as late as six (6) months after the Target Date.',
        "date_index_start": 341,
        "date_index_end": 358,
    },
]


nlpAugmenter = NLPAugmenters()

for index_element, element in enumerate(dataset):

    print("**** paragraph number {} ****".format(index_element))

    first_parts = []
    dates = []
    second_parts = []

    # Take the part of the paragraph and apply the library Data Aug to augment it
    first_sentence = element["paragraph"][: element["date_index_start"]]
    first_parts += nlpAugmenter.synonymAugmenter(
        first_sentence, number_generated_examples=5
    )
    print("[1/8] synonymAug Done")
    first_parts += nlpAugmenter.antonymAugmenter(
        first_sentence, number_generated_examples=5
    )
    print("[2/8] antonymAug Done")
    first_parts += nlpAugmenter.contextualWordEmbsAugmenter(
        first_sentence, number_generated_examples=5
    )
    print("[3/8] contextualWordEmbsAug - insert Done")
    first_parts += nlpAugmenter.contextualWordEmbsAugmenter(
        first_sentence, action="substitute", number_generated_examples=5
    )
    print("[4/8] contextualWordEmbsAug - substitute Done")
    first_parts += nlpAugmenter.generate_text_before_label_with_autoregressive_model(
        first_sentence, number_generated_examples=5
    )
    print("[5/8] xlnetAug Done")
    first_parts += nlpAugmenter.augment_data_using_de_backtranslation(
        first_sentence, first_parts
    )
    print("[6/8] deBacktranslationAug Done")
    first_parts += nlpAugmenter.augment_data_using_ru_backtranslation(
        first_sentence, first_parts
    )
    print("[7/8] ruBacktranslationAug Done")
    first_parts += nlpAugmenter.augment_data_using_ar_backtranslation(
        first_sentence, first_parts
    )
    print("[8/8] arBacktranslationAug Done")
    print(
        "{} new sentence generated for the paragraph's first part. Unique sentences = {}".format(
            len(first_parts), len(list(set(first_parts)))
        )
    )

    # Take the label X and transform it into another type of date
    date = element["paragraph"][element["date_index_start"] : element["date_index_end"]]
    for i in range(len(first_parts)):
        dates.append(transform_date_type(date))
    print("{} new date generated".format(len(dates)))

    # Take the remaining of the paragraph and apply the Data Aug
    second_sentence = element["paragraph"][element["date_index_end"] + 1 :]
    second_parts += nlpAugmenter.synonymAugmenter(
        second_sentence, number_generated_examples=5
    )
    print("[1/8] synonymAug Done")
    second_parts += nlpAugmenter.antonymAugmenter(
        second_sentence, number_generated_examples=5
    )
    print("[2/8] antonymAug Done")
    second_parts += nlpAugmenter.contextualWordEmbsAugmenter(
        second_sentence, number_generated_examples=5
    )
    print("[3/8] contextualWordEmbsAug - insert Done")
    second_parts += nlpAugmenter.contextualWordEmbsAugmenter(
        second_sentence, action="substitute", number_generated_examples=5
    )
    print("[4/8] contextualWordEmbsAug - substitute Done")
    second_parts += nlpAugmenter.generate_text_after_label_with_autoregressive_model(
        second_sentence, number_generated_examples=5
    )
    print("[5/8] xlnetAug Done")
    second_parts += nlpAugmenter.augment_data_using_de_backtranslation(
        second_sentence, second_parts
    )
    print("[6/8] deBacktranslationAug Done")
    second_parts += nlpAugmenter.augment_data_using_ru_backtranslation(
        second_sentence, second_parts
    )
    print("[7/8] ruBacktranslationAug Done")
    second_parts += nlpAugmenter.augment_data_using_ar_backtranslation(
        second_sentence, second_parts
    )
    print("[8/8] arBacktranslationAug Done")
    print(
        "{} new sentence generated for the paragraph's second part. Unique sentences = {}".format(
            len(second_parts), len(list(set(second_parts)))
        )
    )

    # Concatenate the first part, the date and remaining part of each augmented paragraph
    new_paragraphs = []
    paragraphs = []
    min_data = min(len(first_parts), len(second_parts))
    for index in range(min_data):
        new_paragraph = (
            first_parts[index] + " " + dates[index] + " " + second_parts[index]
        )
        paragraphs.append(new_paragraph)
        paragraph_dict = {"paragraph": new_paragraph, "date": dates[index]}
        new_paragraphs.append(paragraph_dict)
    print(
        "{} new paragraphs. Unique paragraphs = {}".format(
            len(paragraphs), len(list(set(paragraphs)))
        )
    )

    # if there are duplicates in first_parts and second_parts lists, we will create random combinations between those two lists to generate at least 200 examples
    if min_data < 200:
        remaining_paragraphs = 200 - min_data
        while remaining_paragraphs > 0:
            random_first_part = random.randint(0, len(first_parts) - 1)
            random_second_part = random.randint(0, len(second_parts) - 1)
            random_date = random.randint(0, len(dates) - 1)

            new_paragraph = (
                first_parts[random_first_part]
                + " "
                + dates[random_date]
                + " "
                + second_parts[random_second_part]
            )

            if new_paragraph not in paragraphs:
                paragraphs.append(new_paragraph)
                paragraph_dict = {
                    "paragraph": new_paragraph,
                    "date": dates[random_date],
                }
                new_paragraphs.append(paragraph_dict)
                remaining_paragraphs -= 1
        print(
            "{} new paragraphs after combination. Unique paragraphs = {}".format(
                len(paragraphs), len(list(set(paragraphs)))
            )
        )

    # Save the augmented data of each paragraph in a json file
    with open(
        "augmentation_paragraph_number_{}.json".format(index_element + 1), "w"
    ) as final:
        json.dump(new_paragraphs, final)
