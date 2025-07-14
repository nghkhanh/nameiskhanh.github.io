---
layout: default
title: 2. Pipeline NLP
nav_order: 2
parent: NLP Course
grand_parent: NLP
permalink: /nlp/course/02_nlp_pipeline/
---

# NLP Pipeline
We have explored some common NLP applications such as Sentiment Analysis, Information Extraction, and Text Summarization, etc... and now we will explore how to build these applications. To build such an application at our organization, we would break the problem into sub-problems and develop a step-by-step procedure, listing all necessary text processing forms. This process, called a pipeline, involves the series of steps needed to build any NLP model. Understanding these common steps is crucial for tackling workplace NLP problems. 

![](images/Pipeline.png)

The above figure outlines the main components of a modern, data-driven NLP system development pipeline. The key stages in this pipeline are:

1. Data acquisition
2. Text cleaning
3. Pre-processing
4. Feature engineering
5. Modeling
6. Evaluation
7. Deployment
8. Monitoring and Model Updating

Developing an NLP system starts with collecting relevant data, followed by text cleaning and pre-processing to standardize the data. Next, feature engineering extracts useful indicators, which are formatted for modeling algorithms. In the modeling and evaluation phase, different models are built and assessed. The best model is then deployed, and its performance is regularly monitored and updated as needed.

In reality, the NLP development process isn't always linear. It often involves revisiting steps like feature extraction, modeling, and evaluation multiple times. There are also loops, especially from evaluation back to pre-processing, feature engineering, and modeling. At the project level, there's a loop from monitoring back to data acquisition.

We'll explore each stage of the NLP pipeline in overview, let's begin with the first step: data acquisition.

## Data Acquisition

Data is crucial for any machine learning system, and often becomes a bottleneck in industrial projects. In this section, we'll explore strategies for gathering relevant data for NLP projects. Sometimes data is readily available, while other times we need to search for it. Text data can be found on various platforms like websites, emails, and social media, but it might not always be in a machine-readable format or relevant to our task. Therefore, understanding our problem is crucial before seeking data. Here are some methods for collecting data:

+ Public dataset: We can search for public datasets to use, if a suitable dataset is found, we can build and evaluate a model. If no appropriate dataset is available, we need to consider other options.
+ Scrape data: We can source relevant data from the internet. Then, this data can be scraped and labeled by human annotators.
+ Data augmentation: Collecting data is effective but time-consuming. Therefore, we can use data augmentation to create text that is syntactically similar to source text data in any NLP problems.

## Text cleaning
At times, our obtained data might not be clean, containing HTML tags, typos, or special characters. Therefore, we need methods to tidy up our text data.

+ Unicode Normalization: When working with text data, we might encounter symbols, emojis, or other special characters. We can change them into text that computers can understand.
+ HTML parsing and cleanup: Involving analyzing HTML code to understand its structure and then removing any unnecessary or incorrect elements (such as HTML tags) to ensure the code is clean and well-formed.
+ Spell checks: Doing basic spell checks to fix common typos and make the text consistent.

Let’s move on to the next step in our pipeline: pre-processing.

## Pre-processing:

Even after cleaning up some text, we still need to pre-process it, especially when crawling data on the Internet. Additionally, NLP software requires text to be split into sentences and words. Pre-processing involves these tasks, as well as removing special characters, digits, and converting text to lowercase if needed. These decisions are made during the pre-processing stage in NLP.

+ Initial steps:
    + Sentence segmentation: Is the process of dividing text into individual sentences.
    + Word tokenization: Is the process of splitting individual sentences that are output from **sentence segmentation** step into separate words.

![](images/initialSteps.png)

+ Regular steps:
    + Stop word removal: Is the process of eliminating common, non-essential words from text.
    + Removing digits/punctuation: Is the process of deleting numbers and punctuation marks from text.
    + Lowercasing: Is the process of converting all letters in text to lowercase.
    + Stemming and lemmatization: Involve reducing words to their base or root forms.

![](images/regularStep.png)

+ Other steps:
    + POS tagging: Is the process of labeling words in a text with their parts of speech, like nouns, verbs, and adjectives.

![](images/PosExample.png)

    + Coreference resolution: Is the process of identifying when different words refer to the same entity in a text.

![](images/coreferenceResolution.png)

    + Parsing tree: Represents the syntactic structure of a string according to a grammar, with nodes denoting constructs and the root representing the start symbol.

![](images/parseTree.png)

For now, we will move on to the next step: feature engineering.

## Feature Engineering:
We've discussed various pre-processing steps and their usefulness. To use these pre-processed texts in machine learning (ML) models, we need feature engineering, which converts text into numerical vectors that ML algorithms can understand. This process is called "text representation".

There are two main approaches to feature engineering:

+ Feature engineering for traditional ML pipelines: Are often manually designed for specific tasks. For instance, in sentiment analysis of product reviews, one might count positive and negative words to predict sentiment. Features are tailored to the task and domain knowledge, and handcrafted features make the model interpretable by showing how each feature influences predictions.

+ Feature engineering for deep learning pipelines: Using pre-processed raw data and learning features from it, usually resulting in better performance. However, DL models lose interpretability because it's hard to explain their predictions. For example, in email spam detection, it's easier to identify which words influenced the decision with handcrafted features, but not with DL models.

There are various feature engineering techniques that we can mention, such as:

+ Bag of Words (BoW): Is a method for text representation where text is converted into a set of words, ignoring grammar and order, and each word's frequency is counted.

![](images/BoW.png)

+ Term Frequency-Inverse Document Frequency (TF-IDF): Is a method for text representation that evaluates the importance of a word in a document based on its frequency in the document and its rarity across all documents.

![](images/TFIDF.png)

+ One-Hot Encoding: Is a method of converting categorical data into binary vectors, where each category is represented by a unique vector with a single '1' and all other positions '0'.

![](images/OneHotVector.png)

+ Word Embeddings (Word2Vec, GloVe, FastText): Are techniques to represent words as dense vectors in a continuous space, capturing semantic relationships between words based on their usage in large text corpora.

![](images/WordEmbedding.png)

All of the techniques will be dicussed deeper in the **Word Embeddings** lesson.

Now, let’s take a look at the next step in the pipeline, which we call modeling.

## Modeling

With our NLP project data in hand and a grasp of necessary cleaning and preprocessing steps, the focus shifts to crafting an effective solution. Initially, simpler methods and rules suffice, particularly with limited data. As our understanding of the problem deepens and more data becomes available, we can gradually introduce complexity to enhance performance. 

When beginning a project with limited data, a heuristic approach can be employed, especially for data collection tasks in ML/DL models. This approach often relies on regular expressions to gather and process data effectively.

After that, we can use machine learning models such as Naive Bayes, Support Vector Machine, Hidden Markov Model, Conditional Random Fields to solves problems.

Last but not least, there has been a significant increase in utilizing neural networks for handling intricate, unorganized data such as language. To tackle this complexity effectively, there's a demand for models with enhanced representation and learning capabilities. Some deep neural networks we can mention are as follows: Recurrent neural networks, Long Short-Term Memory, GRU and the latest are Attention, Transformer, Bert...

## Evaluation
A crucial step in the NLP pipeline is evaluating the model's effectiveness, typically measured by its performance on unseen data. Evaluation metrics vary by NLP task and project phase. During model building and deployment, ML metrics are used, while in production, business metrics are also considered to measure impact.

There are two type of evaluations included:
+ Intrinsic evaluation: These metrics compare the model's output against a test set with known labels to measure how well the model's output matches the labels, such as:
    + RMSE (root mean squared error): Measures the average magnitude of prediction errors in a model, calculated as the square root of the average squared differences between predicted and actual values, use in regression problems such as temperature prediction, stock market price prediction, etc...
    + MAPE (Mean Absolute Percentage Error): Measures the average percentage error between predicted and actual values, indicating the accuracy of a forecasting model.
    + Accuracy: Measures the percentage of correct predictions made by a model out of all predictions, use in classification tasks, such as sentiment classification (multiclass), spam emails detection, etc ...
    + Precision: Measures the percentage of true positive results out of all positive predictions made by the model, use in classification tasks
    + Recall: Measures the percentage of true positive results out of all actual positive instances in the dataset, use in classification tasks.
    + F1 score: Is a metric that combines precision and recall into a single score, representing the harmonic mean of both, use in classification tasks.
    + AUC (Area Under the Curve): Measures the performance of a binary classification model, representing the ability to distinguish between classes. It's typically used with the ROC curve, use in classification tasks.
    + BLEU (Bilingual Evaluation Understudy): Is a metric used to evaluate the quality of machine-translated text by comparing it to one or more reference translations. It measures how many words overlap between the generated and reference translations.
    + ROUGE: (Recall-Oriented Understudy for Gisting Evaluation): Is a metric used to evaluate text summaries by comparing them to reference summaries, focusing on the overlap of n-grams, word sequences, and word pairs.

+ Extrinsic evaluation: Measures the final objective, such as the time users spend dealing with spam emails.


## Deployment

In real-world situations, NLP systems are often integrated into larger systems, such as email spam filters. After completing the processing, modeling, and evaluation steps, deploying the final solution is crucial. This involves integrating the NLP module into the larger system and ensuring its scalability and compatibility with input and output pipelines in production environments.

The NLP module is commonly deployed as a web service, where it functions by receiving text inputs and providing categorized outputs, like spam or non-spam for emails. This service processes emails in real-time, aiding decisions on email handling. For tasks like batch processing, the NLP module might be integrated into broader task queues, as demonstrated in platforms like Google Cloud or AWS.

## Monitoring and model updating
### Monitoring
Similar to other software projects, thorough testing is essential before deploying any NLP model, and continuous monitoring is crucial post-deployment. However, monitoring NLP projects requires special attention to ensure the model's outputs remain meaningful over time. This includes regular checks on model behavior, especially if the model undergoes frequent training updates. Utilizing performance dashboards displaying model parameters and key indicators helps in this regard.

### Model Updating
Once the model is deployed and we start gathering new data, we’ll iterate the model based on this new data to stay current with predictions.

## Conclusion

In this lesson, we explored the Natural Language Processing (NLP) pipeline, from data acquisition to monitoring and model updating. Understanding this pipeline is crucial for building effective NLP applications. 

## References

+ Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, and Harshit Surana, Practical natural language processing : a comprehensive guide to building real-world NLP sysems. Sebastopol, Ca O’reilly Media, 2020.
+ “Natural Language Processing (NLP) Pipeline,” GeeksforGeeks, Jun. 01, 2023. https://www.geeksforgeeks.org/natural-language-processing-nlp-pipeline/
+ “Behind the pipeline - Hugging Face NLP Course,” huggingface.co. https://huggingface.co/learn/nlp-course/en/chapter2/2 (accessed Jul. 07, 2024).