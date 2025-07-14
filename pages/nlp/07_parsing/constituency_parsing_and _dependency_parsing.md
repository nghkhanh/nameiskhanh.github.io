---
layout: default
title: 7. Parsing
nav_order: 7
parent: NLP
permalink: /nlp/07_parsing/
---

# Constituency Parsing and Dependency Parsing

## Introduction

In computational linguistics, parsing involves creating a parse tree that shows the syntactical structure of a sentence. A parse tree illustrates the relationships between words and phrases based on formal grammar.

The two main parsing methods are constituency and dependency parsing, each producing different types of trees due to their distinct grammatical assumptions. Both methods aim to extract syntactic information.

## Constituency Parsing

A constituency parse tree, based on context-free grammars, divides a sentence into sub-phrases (constituents) that belong to specific grammatical categories.

In English, phrases like "a cat," "a cat on the table," and "the nice cat" are noun phrases, while "play football" and "go to school" are verb phrases.

Grammar specifies how to construct valid sentences using a set of rules. For instance, the rule VP → V NP indicates that a verb phrase (VP) can be formed by combining a verb (V) with a noun phrase (NP).

These rules can be used both to generate valid sentences and to analyze a sentence's syntactical structure according to the grammar.

An example of a constituency parse tree:

![](images/parseTree.png)

A constituency parse tree always has the sentence's words as its terminal nodes. Each word typically has a parent node showing its part-of-speech tag, like noun, verb or adjective, though this detail might be omitted in some representations.

Non-terminal nodes in a constituency parse tree represent the sentence's constituents, typically including verb phrases (VP), noun phrases (NP), and prepositional phrases (PP).

In this example, at the first level below the root, our sentence has been split into a noun phrase, made up of the single word “The cat”, and a verb phrase, “sat on the table”. This means that the grammar contains a rule like S → NP VP, meaning that a sentence can be created with the concatenation of a noun phrase and a verb phrase. Similarly, the noun phrase and the verb phrase are divided into smaller parts, which also maps to another rule in the grammar.

In summary, constituency parsing uses context-free grammar to create hierarchical trees that represent a sentence's syntax, dividing it into its phrasal components.

### Applications of Constituency Parsing
Constituency parsing identifies sentence parts (noun phrases, verbs, clauses) and groups them into a tree structure showing their grammatical relationships.

The following are some of the applications of constituency parsing:

1. Natural Language Processing (NLP): It is used in NLP tasks like text summarization, machine translation, question answering, and text classification.
2. Information Retrieval: It extracts information from large texts and indexes it for efficient retrieval.
3. Text-to-Speech: It aids in creating human-like speech by understanding text grammar and structure.
4. Sentiment Analysis: It helps understand if a text is positive, negative, or neutral by figuring out the feelings conveyed by its parts.

## Dependency Parsing

Unlike constituency parsing, dependency parsing doesn't use phrasal constituents or sub-phrases. Instead, it represents sentence structure through dependencies between words, shown as directed, typed edges in a graph.

A dependency parse tree is a graph G = (V, E) where the vertices V are the words in the sentence, and the edges E connect pairs of words. The graph must meet three conditions:

+ There must be one root node that has no incoming edges.
+ For each node v in V, there must be a path from the root R to v.
+ Each node except the root must have exactly 1 incoming edge.

Each edge in E has a type that specifies the grammatical relationship between the two connected words.

An example of a dependency parse tree:

![](images/dependencyParsing.png)

The result is quite different because this method uses the verb of the sentence as the tree's root, with the edges between words showing their relationships.

For example, the word “sat” has an outgoing edge of type subj to the word “cat”, meaning that “cat” is the subject of the verb “sat”. In this case, we say that “cat” depends on “sat”.

### Applications of Dependency Parsing
Dependency parsing analyzes sentence structure by identifying word dependencies and representing them as a directed graph.

The following are some of the applications of dependency parsing:
1. Named Entity Recognition (NER): It helps identify and classify named entities like people, places, and organizations in a text.
2. Part-of-Speech (POS) Tagging: It identifies and classifies each word's part of speech in a sentence, such as nouns, verbs, and adjectives.
3. Machine Translation: It aids in translating sentences by analyzing word dependencies and generating corresponding dependencies in the target language.
4. Text Generation: It generates text by analyzing word dependencies and creating new words that fit the structure.

## Constituency Parsing and Dependency Parsing

| Characteristic | Constituency Parsing | Dependency Parsing |
| --- | --- | --- |
| `Tree Structure` | Constituency parsing creates a hierarchical tree of nested phrases. | Dependency parsing creates a flatter tree where each word points to its head, focusing on word-to-word relationships. |
| `Focus` | Constituency parsing focuses on phrase structure and hierarchy. | Dependency parsing focuses on word relationships and grammatical functions. |
| `Complexity` | Constituency parsing can be more complex due to nested structures. | Dependency parsing is often simpler and more intuitive for representing direct word relationships. |

## Conclusion

In this lesson, we covered Constituency Parsing and Dependency Parsing, two essential techniques for analyzing the syntactic structure of sentences. Constituency Parsing breaks down sentences into sub-phrases, revealing their hierarchical structure, while Dependency Parsing focuses on the relationships between individual words.

Understanding these parsing methods enables us to gain deeper insights into sentence structure and meaning. 

## References

+ “Constituency Parsing and Dependency Parsing,” GeeksforGeeks, Jan. 26, 2023. https://www.geeksforgeeks.org/constituency-parsing-and-dependency-parsing/
+ F. Elia, “Constituency vs Dependency Parsing | Baeldung on Computer Science,” www.baeldung.com, Jun. 17, 2020. https://www.baeldung.com/cs/constituency-vs-dependency-parsing
+ ZenTM️S., “Syntactic Parsing practices in NLP: Constituency and Dependency Parsing,” Plain Simple Software, Oct. 10, 2022. https://medium.com/plain-simple-software/syntactic-parsing-practices-in-nlp-constituency-and-dependency-parsing-43f79244b2af
