---
layout: default
title: 3. Text cleaning
nav_order: 3
parent: NLP Course
grand_parent: NLP
permalink: /nlp/course/03_text_cleaning/
---

# Text Extraction and Cleanup
We discussed some techniques for cleaning raw text in the NLP Pipeline, but it's crucial to emphasize that this step is vital because "garbage in, garbage out." Therefore, we will delve deeper into this step and explore common methods for text cleaning.

Text extraction and cleanup involve removing non-textual information like markup and metadata from raw text and converting it to the needed format. This process varies based on the data format, such as PDFs, HTML, or continuous data streams. For example, the HTML data format as shown in the image below.

![](images/html.png)

## HTML Parsing and Cleanup
Suppose we're building a forum search engine for programming questions using Stack Overflow as a source. To extract question and answer pairs, we can leverage the HTML tags specific to questions and answers on the site. Instead of writing our own HTML parser, we can use libraries like Beautiful Soup and Scrapy. Here's a code example using Beautiful Soup to extract question and answer pairs from a Stack Overflow page.

```python
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import requests
myurl = "https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python"
req = Request(myurl, headers={'User-Agent' : "Magic Browser"}) 
html = urlopen(req).read()
soupified = BeautifulSoup(html, "html.parser")
question = soupified.find("div", {"id": "question-header"})
questiontext = question.find("a", {"class": "question-hyperlink"})
print("Question: \n", questiontext.get_text().strip())
answer = soupified.find("div", {"class": "answercell post-layout--right"})
answertext = answer.find("div", {"class": "s-prose js-post-body"})
print("Best answer: \n", answertext.get_text().strip())
```

We're using our understanding of HTML structure to extract the needed information. This code produces the following output:

```python
Question: 
 How do I get the current time in Python?
Best answer: 
 Use datetime:
>>> import datetime
>>> now = datetime.datetime.now()
>>> now
datetime.datetime(2009, 1, 6, 15, 8, 24, 78915)
>>> print(now)
2009-01-06 15:08:24.789150

For just the clock time without the date:
>>> now.time()
datetime.time(15, 8, 24, 78915)
>>> print(now.time())
15:08:24.789150


To save typing, you can import the datetime object from the datetime module:
>>> from datetime import datetime

Then remove the prefix datetime. from all of the above.
```

In this example, we needed to extract a specific question and answer. For other tasks, like extracting postal addresses from web pages, we might first get all the text from the page. HTML libraries usually have functions to remove HTML tags and return the text content. However, this can result in noisy output, including unwanted JavaScript. To avoid this, we should extract content from tags that usually contain text.

## Unicode Normalization
While cleaning up HTML tags in our code, we may come across various Unicode characters, such as symbols and emojis. 

![](images/Icon.png)

This above image shows some examples of these characters.

To handle non-text symbols and special characters, we use Unicode normalization, converting visible text into a binary format for storage, known as text encoding. Ignoring encoding can cause processing errors later in the pipeline.

Different operating systems have various default encoding schemes. When dealing with multilingual or social media text, it's often necessary to convert between these schemes during text extraction. Here's an example of Unicode handling:

```python
text = "I love 🍕 ! Shall we book a 🚙 to pizza?"
text = text.encode("utf-8")
print(text)
```

which outputs:

```python
b'I love \xf0\x9f\x8d\x95 ! Shall we book a \xf0\x9f\x9a\x99 to pizza?'
```

The processed text is now machine-readable and ready for use in downstream pipelines.

## Regex or Regular Expression
Regular expressions serve as useful tools for recognizing particular patterns within text strings. For example, if our data contains phone numbers, email addresses, or URLs, we can identify them using regular expressions. We have the flexibility to decide whether to keep or remove these identified text patterns based on our requirements. Here is an example of using Regex to extract informative content.

```python
import re 
text = """<Hello everyone>  
#Let's learning together with Robusto AI
url <https://nami.ai/>,  
email <dunghoang@gmail.com> 
"""
def clean_text(text): 
    # remove HTML TAG 
    html = re.compile('[<,#*?>]') 
    text = html.sub(r'',text) 
    # Remove urls: 
    url = re.compile('url https?://\S+|www\.S+') 
    text = url.sub(r'',text) 
    # Remove email id: 
    email = re.compile('email [A-Za-z0-2]+@[\w]+.[\w]+') 
    text = email.sub(r'',text) 
    return text 
print(clean_text(text))
```

which outputs:

```python
Hello everyone  
Let's learning together with Robusto AI
```

The processed text is now solely containing needed information.

## Spelling Correction
In today's fast-paced digital environment, errors in spelling are common due to quick typing and mistakes resulting from pressing the wrong keys. This issue is widespread across various platforms such as search engines, mobile-based text chatbots, and social media. Despite addressing HTML tags and Unicode characters, spelling errors persist as a distinct challenge, potentially affecting the linguistic comprehension of the data. Additionally, shorthand language used in social media microblogs can complicate language processing and context comprehension.

While shorthand language is commonly used in chat interfaces, unintentional typing errors, known as fat-finger problems, are frequent in search engines. Although we acknowledge this issue, there isn't a foolproof solution yet. However, efforts can be made to minimize its impact. Microsoft has introduced a REST API that offers potential spell-checking capabilities, which can be accessed using Python.

**Note:** For more details, let's visit [**Microsoft tutorials**](https://learn.microsoft.com/en-us/previous-versions/azure/cognitive-services/Bing-Spell-Check/quickstarts/python).

## Conclusion

In this lesson, we focused on text cleaning, a vital step in the NLP pipeline. We learned how to parse HTML when crawling text from the Internet, handle misspellings, normalize text, and use regex to capture valuable information. Proper text cleaning ensures that our data is accurate and ready for analysis, significantly improving model performance.


## References

+ Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, and Harshit Surana, Practical natural language processing : a comprehensive guide to building real-world NLP sysems. Sebastopol, Ca O’reilly Media, 2020.
+ “Text Cleaning for NLP: A Tutorial,” MonkeyLearn Blog, May 31, 2021. https://monkeylearn.com/blog/text-cleaning/





