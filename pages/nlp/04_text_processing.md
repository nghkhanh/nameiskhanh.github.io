---
layout: default
title: 4. Text Processing
nav_order: 4
parent: NLP
# grand_parent: NLP
permalink: /nlp/04_textprocessing/
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


## Bước khởi đầu (Preliminaries)

Phần mềm NLP thường bắt đầu bằng việc tách văn bản thành các câu (sentence segmentation) và sau đó thành các từ (word tokenization). Mặc dù các nhiệm vụ này có vẻ đơn giản, nhưng thực tế cần chú ý đặc biệt, như sẽ trình bày ở các phần tiếp theo.


### Tách câu (Sentence segmentation)
Thông thường, ta có thể tách câu bằng cách chia văn bản tại dấu chấm câu hoặc dấu hỏi. Tuy nhiên, các từ viết tắt, danh xưng (Dr., Mr.) hoặc dấu ba chấm (...) có thể làm phức tạp quy tắc này. May mắn là hầu hết các thư viện NLP như Natural Language Tool Kit (NLTK) đã có giải pháp cho các vấn đề này. Ví dụ dưới đây minh họa cách sử dụng NLTK để tách câu và tách từ với một tài liệu mẫu:

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


### Tách từ (Word tokenization)
Tương tự như tách câu, tách từ có thể bắt đầu bằng quy tắc đơn giản: chia văn bản tại các dấu câu. Thư viện NLTK hỗ trợ thực hiện điều này. Ví dụ:

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


Mặc dù các giải pháp có sẵn đáp ứng hầu hết nhu cầu và bao gồm các bộ tách từ, tách câu, nhưng chúng không hoàn hảo. Ví dụ, NLTK có thể tách "Mr. Jack O’Neil" thành ba token: "O", "’", và "Neil". Nó cũng có thể tách sai "$10,000" và "€1000". Ngoài ra, với tweet, nó có thể tách hashtag thành các token riêng biệt.

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


Trong các trường hợp này, cần xây dựng bộ tách từ (tokenizer) tùy chỉnh. Sau khi tách câu, ta sẽ thực hiện tách từ.


## Các bước thường gặp (Frequent Steps)
Trong pipeline NLP, tiền xử lý thường bao gồm loại bỏ từ dừng (stop word)—các từ phổ biến như "a", "an", "the"... vốn không giúp ích cho việc phân loại nội dung. Ví dụ, khi phân loại bài báo thành chính trị, thể thao, kinh doanh..., các từ này không hữu ích. Không có danh sách từ dừng chuẩn, và chúng có thể thay đổi tùy ngữ cảnh. Ví dụ, "news" có thể là từ dừng khi phân loại bài báo nhưng không phải khi trích xuất thông tin từ thư mời làm việc.

Trong một số trường hợp, việc phân biệt chữ hoa/thường không ảnh hưởng đến nhiệm vụ, do đó văn bản thường được chuyển về chữ thường (lowercase). Loại bỏ dấu câu và số cũng là bước phổ biến trong các bài toán NLP như phân loại văn bản, truy xuất thông tin, phân tích mạng xã hội.


### Rút gọn từ và chuẩn hóa từ gốc (Stemming and lemmatization)
#### Rút gọn từ (Stemming)
Stemming là quá trình loại bỏ hậu tố của từ để đưa về dạng gốc chung. Ví dụ, "car" và "cars" đều thành "car". Quy tắc stemming có thể không luôn chính xác về mặt ngôn ngữ, nhưng được ứng dụng rộng rãi trong công cụ tìm kiếm để khớp truy vấn với tài liệu liên quan, và trong phân loại văn bản để giảm số lượng đặc trưng cho mô hình học máy.

Dưới đây là ví dụ sử dụng kỹ thuật stemming phổ biến là Porter Stemmer qua NLTK:

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


Quá trình stemming chuyển "cars" thành "car" và "revolution" thành "revolut", điều này không chính xác về mặt ngôn ngữ. Tuy nhiên, với công cụ tìm kiếm thì không ảnh hưởng nhiều, nhưng trong một số trường hợp, cần giữ đúng dạng ngôn ngữ. Khi đó, ta dùng lemmatization—quá trình tương tự nhưng chính xác hơn về mặt ngôn ngữ học.


#### Chuẩn hóa từ gốc (Lemmatization)
Lemmatization là quá trình đưa các dạng khác nhau của một từ về dạng gốc (lemma), tương tự stemming nhưng có sự khác biệt. Ví dụ, stemming giữ nguyên "better", còn lemmatization chuyển thành "good". Lemmatization đòi hỏi hiểu biết sâu hơn về ngôn ngữ học, và việc xây dựng bộ lemmatizer hiệu quả vẫn là thách thức trong nghiên cứu NLP.

Dưới đây là ví dụ sử dụng lemmatizer dựa trên WordNet với NLTK:

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


Và đoạn mã sau minh họa lemmatizer sử dụng SpaCy:

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


NLTK trả về "good", còn spaCy trả về "better well", cả hai đều được coi là đúng. Lemmatization cần phân tích ngôn ngữ sâu hơn, thường chậm hơn stemming và chỉ dùng khi thực sự cần thiết. Việc chọn lemmatizer linh hoạt, có thể dùng NLTK hoặc spaCy tùy framework tiền xử lý khác để đảm bảo nhất quán toàn pipeline.

Đây là các bước tiền xử lý phổ biến, nhưng chưa phải tất cả. Tùy đặc điểm dữ liệu, có thể cần thêm các bước tiền xử lý khác. Hãy cùng tìm hiểu một số bước bổ sung.


## Các bước tiền xử lý khác (Other Pre-Processing Steps)
Đến đây, chúng ta đã đề cập các bước tiền xử lý phổ biến trong NLP, giả định làm việc với văn bản tiếng Anh chuẩn. Nhưng nếu xử lý các loại văn bản khác thì sao? Khi đó, cần thêm các bước tiền xử lý bổ sung, ví dụ:

### Chuẩn hóa văn bản (Text normalization)
Giả sử ta phân tích bài đăng mạng xã hội để phát hiện sự kiện, nơi ngôn ngữ sử dụng khác biệt nhiều so với văn viết chuẩn như báo chí. Văn bản mạng xã hội thường có nhiều biến thể: chính tả khác nhau, viết tắt, kiểu chữ đa dạng, định dạng số khác nhau... Để xử lý hiệu quả, ta cần chuẩn hóa văn bản về một dạng thống nhất (text normalization): chuyển về chữ thường/hoa, thay số bằng chữ, mở rộng viết tắt... Các công cụ như SpaCy có từ điển ánh xạ các biến thể về một dạng chuẩn.

### Nhận diện ngôn ngữ (Language detection)
Phần lớn nội dung trên web không phải tiếng Anh. Ví dụ, khi thu thập đánh giá sản phẩm, ta thường gặp nhiều ngôn ngữ khác nhau. Điều này gây khó khăn cho pipeline NLP thiết kế cho tiếng Anh. Để giải quyết, nhận diện ngôn ngữ là bước đầu tiên trong pipeline. Sau khi xác định ngôn ngữ, các bước tiếp theo sẽ được điều chỉnh phù hợp.


## Xử lý nâng cao (Advanced Processing)

Giả sử bạn cần xây dựng hệ thống nhận diện tên người và tổ chức trong tập tài liệu lớn của công ty. Các bước tiền xử lý thông thường có thể không áp dụng được. Để nhận diện tên, ta cần gán nhãn từ loại (POS tagging) để xác định danh từ riêng. Vậy làm sao tích hợp POS tagging vào pipeline tiền xử lý? Các bộ gán nhãn POS đã được huấn luyện sẵn trong thư viện NLP như NLTK, spaCy, Parsey McParseface Tagger, giúp bạn không phải tự xây dựng từ đầu. Dưới đây là ví dụ sử dụng các hàm tiền xử lý có sẵn trong spaCy:

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


## Kết luận (Conclusion)

Trong bài học này, chúng ta đã tìm hiểu về tiền xử lý văn bản—một bước quan trọng trong pipeline NLP. Chúng ta đã khám phá từ các kỹ thuật cơ bản như tách từ, loại bỏ từ dừng, rút gọn từ, chuẩn hóa từ gốc, loại bỏ số/ký tự đặc biệt, chuyển về chữ thường, đến các kỹ thuật nâng cao như gán nhãn từ loại (POS tagging), phân tích cú pháp (parsing), giải quyết đồng tham chiếu (coreference resolution). Các bước tiền xử lý này rất cần thiết để chuyển văn bản thô thành dạng sạch, có cấu trúc, giúp mô hình học máy xử lý hiệu quả.


## Tài liệu tham khảo (References)

+ Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, and Harshit Surana, Practical natural language processing : a comprehensive guide to building real-world NLP sysems. Sebastopol, Ca O’reilly Media, 2020.
+ “Getting started with Text Preprocessing,” kaggle.com. https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
+ “Text Preprocessing in Python | Set - 1,” GeeksforGeeks, May 24, 2019. https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/




