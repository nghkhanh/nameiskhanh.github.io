---
layout: default
title: 5. Parts-Of-Speech
nav_order: 5
parent: NLP Course
grand_parent: NLP
permalink: /nlp/course/05_pos_tagging/
---

# POS(Parts-Of-Speech) Tagging in NLP

## Introduction

One essential task in Natural Language Processing (NLP) is Parts of Speech (PoS) tagging, which involves assigning grammatical categories like nouns, verbs, adjectives, and adverbs to each word in a text. This helps machines better understand and process human language by improving their grasp of phrase structure and meaning.

PoS tagging is crucial for NLP applications like machine translation, sentiment analysis, and information retrieval. It connects language to machine understanding, allowing for the development of advanced language processing systems and deeper linguistic analysis.

## What is POS Tagging?

PoS tagging in NLP assigns each word in a document a specific part of speech, such as adverb, adjective, or verb. This adds syntactic and semantic information, making it easier to understand the sentence's structure and meaning.

In NLP, PoS tagging is useful for tasks like machine translation, named entity recognition, and information extraction. It helps clarify ambiguous terms and reveals a sentence's grammatical structure.

![](images/pos.png)

## Example of POS Tagging
Consider the sentence: "The big brown capybara is on the street."

![](images/PosExample.png)

This tagging helps machines understand not only individual words but also their connections within phrases, providing valuable insights into grammatical structure. Such data is crucial for various NLP tasks like text summarization, sentiment analysis, and machine translation.

## POS Tagging Pipeline

A pipeline for Part-of-Speech (POS) tagging typically involves several key steps:

![](images/POSTaggingPipeline.png)

+ Text Preprocessing: Clean and preprocess the text by removing punctuation, converting text to lowercase, and tokenizing the text into individual words.

+ Tokenization: Split the text into tokens (words or phrases) that will be analyzed.

+ POS Tagging: Use a POS tagging algorithm or library (such as NLTK, SpaCy, or Stanford NLP) to assign POS tags to each token. This can involve using pre-trained models or training a custom model.

+ Post-Processing: Refine and validate the tagged output, correcting any errors and ensuring consistency.

+ Output: Generate the final tagged text, which can be used for further natural language processing tasks or analysis.

## Types of POS Tagging in NLP
Part-of-Speech (PoS) tagging, which assigns grammatical categories to words in a text, is crucial in Natural Language Processing (NLP). Various PoS tagging methods exist, each with its own approach. 
Here are some common types.

### Rule-Based Tagging

Rule-Based POS Tagging uses predefined linguistic rules and lexicons to assign part-of-speech tags to words in a text. It relies on hand-crafted rules and dictionaries to determine the correct tags based on word morphology and context.

For example, a rule-based POS tagger might assign the "noun" tag to words that end in "‑tion" or "‑ment," identifying typical suffixes for nouns. This method is transparent and interpretable because it doesn't depend on training data.


### Transformation Based Tagging

Transformation-Based Tagging, also known as Brill Tagging, iteratively applies predefined transformation rules to improve initial POS tags assigned by a simple method, refining the tags based on contextual patterns.

For instance, a verb's tag might change to a noun if it follows a determiner like "the." These rules are applied systematically, updating the tags after each transformation.

Transformation Based Tagging can be more accurate than rule-based tagging, especially for complex grammar, but it may need a large set of rules and more computational power for optimal performance.

### Statistical POS Tagging

Statistical part-of-speech (POS) tagging uses probabilistic models and machine learning to assign grammatical categories to words in a text, unlike rule-based tagging which relies on predefined rules and annotated corpora.

Statistical POS tagging algorithms learn the probability of word-tag sequences to capture language patterns. Popular models include CRFs and Hidden Markov Models (HMMs). During training, the algorithm uses labeled examples to estimate the likelihood of a specific tag given the word and its context.

Using the trained model, statistical POS tagging predicts the most likely tags for new text. It is particularly effective for languages with complex grammar, as it handles linguistic ambiguity and captures subtle language patterns well.

#### Hidden Markov Model POS tagging
Hidden Markov Models (HMMs) are used for part-of-speech (POS) tagging in NLP. They are trained on large annotated text corpora to recognize patterns in parts of speech. Using this training, HMMs predict the POS tag for a word based on the probabilities of various tags in its context.

An HMM-based POS tagger uses states for potential POS tags and transitions between them. It learns the probabilities of these transitions and word emissions during training. When tagging new text, it applies the Viterbi algorithm to determine the most likely sequence of POS tags based on these probabilities.

HMMs are widely used in NLP for handling complex sequential data, but their performance depends on the quality and amount of annotated training data.

## Parts-of-Speech implementation
We will leverage **NLTK** nad **Spacy** libraries to implement Parts-of-Speech tagging in **Python**

### Implementation of Parts-of-Speech tagging using NLTK in Python
#### Installing packages
```python
!pip install nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

#### Import packages
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
```

#### Implementation
```python
# Sample text
text = "The big brown capybara is on the street."
words = word_tokenize(text)

# Performing PoS tagging
pos_tags = pos_tag(words)
 
# Displaying the PoS tagged result in separate lines
print("Original Text:")
print(text)
 
print("\nPoS Tagging Result:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")
```

which outputs:

```python
Original Text:
The big brown capybara is on the street.


from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Sample text
text = "The big brown capybara is on the street."
words = word_tokenize(text)

# Performing PoS tagging
pos_tags = pos_tag(words)


Original Text:
The big brown capybara is on the street.

PoS Tagging Result:
The: DT
big: JJ
brown: NN
capybara: NN
is: VBZ
on: IN
the: DT
street: NN
.: .
```

Firstly, we employ the **word_tokenize** method to tokenize the input text into words. 

Then, utilize the **pos_tag** function from **NLTK** to conduct part-of-speech tagging on the tokenized words.

Finally, we display the original text and the resulting POS tags separately, exhibiting each word with its associated part-of-speech tag on different lines.

### Implementation of Parts-of-Speech tagging using Spacy in Python
#### Installing packages
```python
!pip install spacy
!python -m spacy download en_core_web_sm
```

#### Import packages 
```python
#importing libraries 
import spacy
```

#### Implementation
```python 
# Load the English language model
nlp = spacy.load("en_core_web_sm")
 
# Sample text
text = "The big brown capybara is on the street."
 
# Process the text with SpaCy
doc = nlp(text)
 
# Display the PoS tagged result
print("Original Text: ", text)
print("PoS Tagging Result:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
```
which outputs:

```python
Original Text:  The big brown capybara is on the street.
PoS Tagging Result:
The: DET
big: ADJ
brown: ADJ
capybara: NOUN
is: AUX
on: ADP
the: DET
street: NOUN
.: PUNCT
```

We utilize the **SpaCy** library and load the English language model "en_core_web_sm" with **spacy.load("en_core_web_sm")**.
Next, the sample text is processed using the loaded model to generate a **Doc** object that is containing linguistic annotations.
Then, we print the original text and iterate through the processed Doc tokens, showcasing each token's text and its corresponding part-of-speech tag (token.pos_).

**Note**: POS tagging models in the NLTK and spaCy libraries rely on statistical approaches, particularly employing trained Hidden Markov Models (HMM).

## Conclusion

In this lesson, we delved into Part-of-Speech (POS) tagging, a fundamental task in NLP that involves labeling words in a sentence with their corresponding parts of speech, such as nouns, verbs, adjectives, and more as well as various methods of implementing POS tagging.

With this understanding, we can now enhance our text analysis by accurately identifying the grammatical structure of sentences.

## References

+ Geeksforgeeks, “NLP | Part of Speech - Default Tagging,” GeeksforGeeks, Jan. 28, 2019. https://www.geeksforgeeks.org/nlp-part-of-speech-default-tagging/
+ S. Mudadla, “What is Parts of Speech (POS) Tagging Natural Language Processing?In,” Medium, Nov. 09, 2023. https://medium.com/@sujathamudadla1213/what-is-parts-of-speech-pos-tagging-natural-language-processing-in-2b8f4b07b186