---
layout: default
title: 4. Text Processing
nav_order: 4
parent: NLP Course
grand_parent: NLP
permalink: /nlp/course/04_textprocessing/
---

# Text Preprocessing

## Introduction

Why do we need to pre-process text even after initial cleanup? 

When processing text from Wikipedia pages for biographical information, our crawled data often contains irrelevant HTML and boilerplate content. Text extraction removes this, leaving us with plain text. However, NLP software requires text to be separated into sentences and words. We also might need to remove special characters, digits, and standardize case. These tasks are handled during the pre-processing step of the NLP pipeline. Here are some common pre-processing steps used in NLP software:

**Preliminaries**:
    Sentence segmentation and word tokenization.

**Frequent steps**:
    Stop word removal, stemming and lemmatization, removing digits/punctuation, lowercasing, etc.

**Other steps**:
    Normalization, language detection, code mixing, transliteration, etc.

**Advanced processing**:
    POS tagging, parsing, coreference resolution, etc.

Not every NLP pipeline includes all steps, but the first two are usually present. Let's explore what each step involves.

## Preliminaries

NLP software usually starts by breaking text into sentences (sentence segmentation) and then into words (word tokenization). While these tasks may seem simple, they require special attention, as we'll explore in the next subsections.

### Sentence segmentation
We can generally segment sentences by breaking text at full stops and question marks. However, abbreviations, titles (Dr., Mr.), or ellipses (...) can complicate this rule. Fortunately, most NLP libraries, like the Natural Language Tool Kit (NLTK), already have solutions for these issues. The code example below demonstrates using NLTK for sentence and word splitting with the example document as input:

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

mytext = '''In the previous chapter, we saw examples of some common NLP \
applications that we might encounter in everyday life. If we were asked to \
build such an application, think about how we would approach doing so at our \
organization. We would normally walk through the requirements and break the \
problem down into several sub-problems, then try to develop a step-by-step \
procedure to solve them. Since language processing is involved, we would also \
list all the forms of text processing needed at each step. This step-by-step \
processing of text is known as pipeline. It is the series of steps involved in \
building any NLP model. These steps are common in every NLP project, so it \
makes sense to study them in this chapter. Understanding some common procedures \
in any NLP pipeline will enable us to get started on any NLP problem encountered \
in the workplace. Laying out and developing a text-processing pipeline is seen \
as a starting point for any NLP application development process. In this \
chapter, we will learn about the various steps involved and how they play \
important roles in solving the NLP problem and we’ll see a few guidelines \
about when and how to use which step. In later chapters, we’ll discuss \
specific pipelines for various NLP tasks (e.g., Chapters 4–7).
'''

my_sentences = sent_tokenize(mytext)
```

which outputs:

```python
['In the previous chapter, we saw examples of some common NLP applications that we might encounter in everyday life.',
 'If we were asked to build such an application, think about how we would approach doing so at our organization.',
 'We would normally walk through the requirements and break the problem down into several sub-problems, then try to develop a step-by-step procedure to solve them.',
 'Since language processing is involved, we would also list all the forms of text processing needed at each step.',
 'This step-by-step processing of text is known as pipeline.',
 'It is the series of steps involved in building any NLP model.',
 'These steps are common in every NLP project, so it makes sense to study them in this chapter.',
 'Understanding some common procedures in any NLP pipeline will enable us to get started on any NLP problem encountered in the workplace.',
 'Laying out and developing a text-processing pipeline is seen as a starting point for any NLP application development process.',
 'In this chapter, we will learn about the various steps involved and how they play important roles in solving the NLP problem and we’ll see a few guidelines about when and how to use which step.',
 'In later chapters, we’ll discuss specific pipelines for various NLP tasks (e.g., Chapters 4–7).']
```

### Word tokenization
Just like sentence tokenization, word tokenization can start with a simple rule: splitting text at punctuation marks. The NLTK library helps us do this. For example:

```python
for sentence in my_sentences:
   print(sentence)
   print(word_tokenize(sentence))
```

For the first sentence, the output is printed as follows:

```python
In the previous chapter, we saw examples of some common NLP applications that we might encounter in everyday life.
['In', 'the', 'previous', 'chapter', ',', 'we', 'saw', 'examples', 'of', 'some', 'common', 'NLP', 'applications', 'that', 'we', 'might', 'encounter', 'in', 'everyday', 'life', '.']
```

While readily available solutions work for most needs and include tokenizers and sentence splitters, they aren't perfect. For instance, NLTK might split "Mr. Jack O’Neil" into three tokens: "O", "’", and "Neil". It can also incorrectly tokenize "$10,000" and "€1000". Additionally, for tweets, it might split hashtags into separate tokens.

```python
print(word_tokenize("Mr. Jack O’Neil"))
print(word_tokenize("$10,000"))
print(word_tokenize("#robusto.ai"))
```

which will output respectively:

```python
['Mr.', 'Jack', 'O', '’', 'Neil']
['$', '10,000']
['#', 'robusto.ai']
```

In such cases, custom tokenizers are necessary. After sentence tokenization, we'll demonstrate word tokenization.

## Frequent Steps
In an NLP pipeline, pre-processing often includes removing stop words—common words like "a," "an," "the," etc., which don't help categorize content. For instance, in categorizing news articles into politics, sports, business, and other categories, these words aren't useful. There's no standard list of stop words, and they can vary by context. For example, "news" might be a stop word for categorizing articles but not for extracting information from job offer letters.

In certain situations, the case of letters might not affect the task. Therefore, text is often converted to lowercase. Removing punctuation and numbers is another typical step in various NLP tasks like text classification, information retrieval, and social media analytics. 

### Stemming and lemmatization
#### Stemming
Stemming involves removing word suffixes to simplify words to a common base form. For instance, "car" and "cars" both become "car." While stemming rules may not always yield linguistically accurate forms, it's widely applied in search engines to match queries with relevant documents and in text classification to streamline feature spaces for machine learning models.

Below is a code example demonstrating the application of a widely used stemming technique known as Porter Stemmer through NLTK:

```python
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
word1, word2 = "cars", "revolution" 
print(stemmer.stem(word1), stemmer.stem(word2))
```

which outputs:

```python
car revolut
```

The stemming process converts "cars" to "car" and "revolution" to "revolut," which isn't linguistically accurate. While this might not impact a search engine's effectiveness, in certain situations, having the correct linguistic form is essential. This is where lemmatization, a process similar to stemming but more linguistically accurate, comes into play.

#### Lemmatization
Lemmatization involves reducing various forms of a word to its base form or lemma, similar to stemming but with distinctions. For instance, while stemming keeps "better" unchanged, lemmatization transforms it into "good." Unlike stemming, lemmatization demands deeper linguistic understanding, and creating effective lemmatizers remains a challenge in ongoing NLP research.

Here's an example of how we can utilize a lemmatizer relying on WordNet, demonstrated using NLTK:

```python
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a")) # a is for adjective
```

which outputs:
```python
good
```

And this code snippet shows a lemmatizer using SpaCy:

```python
import spacy
sp = spacy.load('en_core_web_sm')
token = sp(u'better')
for word in token:
   print(word.text,  word.lemma_)
```

which outputs:
```python
better well
```

NLTK outputs "good," while spaCy outputs "better well," and both are considered correct. Lemmatization, requiring more linguistic analysis, might take longer compared to stemming and is usually employed only when essential. The lemmatizer choice is flexible; we can opt for NLTK or spaCy based on the framework used for other preprocessing steps, aiming for consistency across the pipeline.

These are the typical pre-processing steps, but they don't cover everything. Depending on the data's characteristics, additional pre-processing steps might be necessary. Let's explore some of those additional steps.

## Other Pre-Processing Steps
Until now, we've covered typical pre-processing steps in NLP, assuming we're working with standard English text. But what if we're dealing with different types of text? In such cases, we'll need additional pre-processing steps, which we'll explore next with some examples.
### Text normalization
Imagine we're analyzing social media posts to detect news events, where the language used differs significantly from formal writing, like in newspapers. Social media text often contains variations like different spellings, abbreviated forms, varied casing, and diverse formats for numbers. To effectively process such data, we aim to standardize text into a unified format, a process known as text normalization. This involves converting text to lowercase or uppercase, replacing digits with their textual equivalents, expanding abbreviations, and more. Tools like SpaCy offer dictionaries mapping variant spellings to a single standard form.

### Language detection
Much of the content on the web exists in languages other than English. For instance, when gathering product reviews online, we often encounter reviews in various languages. This poses a challenge for NLP pipelines designed for English text. To address this, language detection serves as the initial step in the pipeline. After identifying the language, subsequent pipeline steps can be tailored accordingly to handle text in the detected language.

## Advanced Processing

Let's say we're tasked with creating a system to recognize names of people and organizations in a large collection of company documents. The usual pre-processing steps we've talked about may not apply here. To spot names, we need POS tagging, which helps identify proper nouns. But how do we incorporate POS tagging into our project's pre-processing phase? Pre-trained POS taggers are readily available in NLP libraries like NLTK, spaCy, and Parsey McParseface Tagger. This allows us to avoid creating our own POS-tagging solutions. Below is an example code snippet showcasing several pre-built pre-processing functions using the spaCy NLP library.

```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'Charles Spencer Chaplin was born on 16 April 1889 toHannah Chaplin (born Hannah Harriet Pedlingham Hill) and Charles Chaplin Sr')
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_alpha, token.is_stop)
```

which outputs:
```python
Charles Charles PROPN True False
Spencer Spencer PROPN True False
Chaplin Chaplin PROPN True False
was be AUX True True
born bear VERB True False
on on ADP True True
16 16 NUM False False
April April PROPN True False
1889 1889 NUM False False
toHannah toHannah PROPN True False
Chaplin Chaplin PROPN True False
( ( PUNCT False False
born bear VERB True False
Hannah Hannah PROPN True False
Harriet Harriet PROPN True False
Pedlingham Pedlingham PROPN True False
Hill Hill PROPN True False
) ) PUNCT False False
and and CCONJ True True
Charles Charles PROPN True False
Chaplin Chaplin PROPN True False
Sr Sr PROPN True False
```

## Conclusion

In this lesson, we covered text preprocessing, a critical step in the NLP pipeline. We explored from basic techniques, such as tokenization, stop word removal, stemming and lemmatization, removing digits/punctuation, lowercasing, to advance techniques , such as POS tagging, parsing, coreference resolution. These preprocessing steps are essential for converting raw text into a clean, structured format that can be effectively used by machine learning models. 

## References

+ Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, and Harshit Surana, Practical natural language processing : a comprehensive guide to building real-world NLP sysems. Sebastopol, Ca O’reilly Media, 2020.
+ “Getting started with Text Preprocessing,” kaggle.com. https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
+ “Text Preprocessing in Python | Set - 1,” GeeksforGeeks, May 24, 2019. https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/




