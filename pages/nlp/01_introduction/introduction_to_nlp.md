---
layout: default
title: 1. Giới thiệu NLP
nav_order: 1
parent: NLP
permalink: /nlp/01_introduction/
---


# Xử lý ngôn ngữ tự nhiên trong thực tế

## Giới thiệu

NLP, viết tắt của Xử lý ngôn ngữ tự nhiên, đóng vai trò quan trọng trong nhiều ứng dụng phần mềm hàng ngày. Một số ví dụ tiêu biểu gồm:

1. **Nền tảng Email**: Các dịch vụ như Gmail và Outlook sử dụng NLP để phân loại thư rác, hộp thư ưu tiên và tính năng tự động hoàn thành.
2. **Trợ lý giọng nói**: Apple Siri, Google Assistant, Microsoft Cortana và Amazon Alexa dựa vào NLP để tương tác với người dùng và phản hồi lệnh.
3. **Công cụ tìm kiếm**: Google và Bing sử dụng NLP để hiểu truy vấn, truy xuất thông tin và xếp hạng kết quả.
4. **Dịch máy**: Google Translate và Amazon Translate áp dụng NLP để hỗ trợ giao tiếp và giải quyết các tình huống kinh doanh.

Ngoài ra, còn có các ứng dụng khác của NLP:

1. Phân tích mạng xã hội: Các tổ chức phân tích mạng xã hội để hiểu cảm xúc của khách hàng.
2. Thương mại điện tử: NLP trích xuất thông tin từ mô tả sản phẩm và hiểu đánh giá của người dùng.
3. Công cụ kiểm tra chính tả và ngữ pháp: Các công cụ như Grammarly và kiểm tra chính tả trong trình soạn thảo văn bản dựa vào NLP.

Danh sách này chưa đầy đủ, vì NLP vẫn tiếp tục mở rộng sang nhiều ứng dụng mới. Mục tiêu chính của chúng ta là giới thiệu các khái niệm đằng sau việc xây dựng các ứng dụng này thông qua việc thảo luận các vấn đề và giải pháp NLP khác nhau.


## NLP

Trong nhiều dự án NLP, có những nhiệm vụ cơ bản thường xuyên xuất hiện. Những nhiệm vụ này rất quan trọng và phổ biến. Nếu hiểu rõ chúng, chúng ta sẽ sẵn sàng xây dựng các ứng dụng NLP cho nhiều lĩnh vực khác nhau. Hãy cùng điểm qua các nhiệm vụ này.

### Mô hình ngôn ngữ

Nhiệm vụ này liên quan đến việc dự đoán từ tiếp theo trong một câu dựa vào các từ trước đó. Mục tiêu là hiểu xác suất xuất hiện của các chuỗi từ trong ngôn ngữ. Nó hữu ích cho nhiều ứng dụng như nhận diện giọng nói, dịch thuật và sửa lỗi chính tả.

![](images/languageModeling.png)

### Phân loại văn bản

Nhiệm vụ này là phân loại văn bản vào các nhóm đã định sẵn dựa trên nội dung. Nó được sử dụng rộng rãi trong NLP cho các tác vụ như nhận diện email rác và phân tích cảm xúc.

![](images/textClassification.png)

### Trích xuất thông tin

Nhiệm vụ này là lấy ra các thông tin hữu ích từ văn bản, ví dụ như xác định sự kiện từ email hoặc nhận diện tên người trong các bài đăng mạng xã hội. Một trong những nhiệm vụ phổ biến nhất là Trích xuất thực thể tên (NER).

![](images/NERintro.png)

### Hệ thống hội thoại

Nhiệm vụ này là xây dựng các hệ thống có khả năng giao tiếp bằng ngôn ngữ tự nhiên, ví dụ như các ứng dụng phổ biến như Alexa và Siri.

![](images/dialogSystem.png)

### Tóm tắt văn bản

Nhiệm vụ này tập trung vào việc tạo ra bản tóm tắt ngắn gọn cho các tài liệu dài mà vẫn giữ được nội dung và thông điệp chính.

![](images/textSummarization.png)

### Trả lời câu hỏi

Nhiệm vụ này là xây dựng hệ thống có khả năng tự động trả lời các câu hỏi được đặt ra bằng ngôn ngữ tự nhiên.

![](images/questionAnswering.png)

### Dịch máy

Nhiệm vụ này là dịch văn bản từ ngôn ngữ này sang ngôn ngữ khác, với các công cụ như Google Translate là ví dụ điển hình.

![](images/machineTranlation.png)


## Ngôn ngữ là gì?

Ngôn ngữ là một hệ thống giao tiếp phức tạp được cấu trúc quanh các thành phần như ký tự, từ và câu. Ngôn ngữ học, ngành nghiên cứu về ngôn ngữ, cung cấp những hiểu biết quan trọng để hiểu về NLP. Trong phần này, chúng ta sẽ khám phá các khái niệm ngôn ngữ học chính và sự liên quan của chúng đến các nhiệm vụ NLP. Ngôn ngữ con người gồm các thành phần: âm vị, hình vị, từ vị, cú pháp và ngữ cảnh. Các ứng dụng NLP dựa vào việc hiểu các yếu tố này, từ đơn vị âm thanh cơ bản đến các biểu đạt có ý nghĩa trong ngữ cảnh.

Ngôn ngữ học là ngành nghiên cứu về ngôn ngữ và rất rộng lớn, ở đây chúng ta chỉ giới thiệu một số ý tưởng cơ bản để minh họa vai trò của kiến thức ngôn ngữ học trong NLP.

### Âm vị
Âm vị là đơn vị âm thanh nhỏ nhất trong một ngôn ngữ, bản thân nó không mang ý nghĩa nhưng khi kết hợp lại sẽ tạo thành từ có nghĩa. Tiếng Anh có 44 âm vị. Bảng dưới đây minh họa các âm vị này cùng với ví dụ (bảng này lấy từ sách **Practical Natural Language Processing**).

![](images/phonemes.png)

Âm vị rất quan trọng cho các ứng dụng như nhận diện giọng nói, chuyển giọng nói thành văn bản và chuyển văn bản thành giọng nói.

### Hình vị và từ vị

#### Hình vị
Hình vị là đơn vị nhỏ nhất mang nghĩa của ngôn ngữ, được tạo thành từ các âm vị. Không phải tất cả hình vị đều là từ, nhưng tất cả tiền tố và hậu tố đều là hình vị. Ví dụ, trong từ "multimedia", "multi-" là tiền tố và hình vị bổ nghĩa cho "media".

#### Từ vị
Từ vị là các biến thể của hình vị có cùng ý nghĩa, như "run" và "running". Phân tích hình thái học nghiên cứu các cấu trúc này, là nền tảng cho nhiều nhiệm vụ NLP như tách từ, rút gọn từ, học biểu diễn từ và gán nhãn từ loại.

### Cú pháp
Cú pháp là các quy tắc để xây dựng câu đúng ngữ pháp từ các từ và cụm từ. Thường được biểu diễn bằng cây phân tích cú pháp như hình dưới.

![](images/parseTree.png)

### Ngữ cảnh
Ngữ cảnh là cách các phần khác nhau của ngôn ngữ kết hợp để truyền đạt ý nghĩa, bao gồm tham chiếu dài hạn, kiến thức thế giới và kiến thức thông thường, vượt ra ngoài nghĩa đen của từ. Ý nghĩa của câu có thể thay đổi tùy vào ngữ cảnh vì từ và cụm từ có thể mang nhiều nghĩa, như ví dụ "Ông già" bên dưới. Ngữ cảnh gồm ngữ nghĩa (ý nghĩa trực tiếp của từ và câu) và ngữ dụng (thêm ngữ cảnh bên ngoài và kiến thức thế giới). Các nhiệm vụ NLP phức tạp như phát hiện mỉa mai và tóm tắt văn bản phụ thuộc nhiều vào việc hiểu ngữ cảnh.

![](images/oldman.png)


## Thách thức của NLP
NLP gặp nhiều thách thức do sự mơ hồ và sáng tạo của ngôn ngữ con người. Chúng ta sẽ tìm hiểu các đặc điểm này, bắt đầu từ sự mơ hồ của ngôn ngữ.

### Sự mơ hồ
Sự mơ hồ là sự không chắc chắn về ý nghĩa trong ngôn ngữ. Ví dụ, câu "Đưa cho tôi cái gậy", không rõ "gậy" ở đây là con dơi hay cây gậy chơi thể thao. Chỉ nhìn vào từ thì không đủ thông tin để xác định ý nghĩa, do đó cần biết ngữ cảnh sử dụng. Ý nghĩa phụ thuộc vào ngữ cảnh. Trong câu chuyện giữa các nhà nghiên cứu động vật, "gậy" là con dơi, còn trong bối cảnh thể thao, "gậy" là cây gậy cricket.

### Kiến thức chung
Một phần quan trọng của ngôn ngữ con người là "kiến thức chung", gồm những sự thật mà hầu hết mọi người đều biết. Kiến thức này được ngầm hiểu trong giao tiếp và ảnh hưởng đến ý nghĩa câu mà không cần nói rõ. Ví dụ, "người cắn chó" là điều hiếm gặp, còn "chó cắn người" thì hợp lý vì chúng ta biết con người hiếm khi cắn chó, còn chó thường cắn người. Kiến thức chung này giúp con người hiểu ngôn ngữ, nhưng máy tính lại gặp khó khăn vì thiếu sự hiểu ngầm này. Một thách thức lớn của NLP là mã hóa kiến thức chung của con người vào mô hình máy tính.

### Sự sáng tạo
Ngôn ngữ không chỉ tuân theo quy tắc mà còn có sự sáng tạo, thể hiện qua phong cách, phương ngữ, thể loại và biến thể. Thơ ca là ví dụ điển hình cho sự sáng tạo này. Việc hiểu được sự sáng tạo là thách thức lớn trong cả NLP và AI nói chung.

### Sự đa dạng giữa các ngôn ngữ
Hầu hết các ngôn ngữ không có sự tương đồng hoàn toàn về từ vựng, khiến việc chuyển giao giải pháp NLP giữa các ngôn ngữ trở nên khó khăn. Một giải pháp hiệu quả cho một ngôn ngữ có thể không áp dụng được cho ngôn ngữ khác. Điều này đòi hỏi phải xây dựng giải pháp không phụ thuộc vào ngôn ngữ, điều này rất khó, hoặc phát triển giải pháp riêng cho từng ngôn ngữ, điều này tốn thời gian và công sức.

NLP có nhiều thách thức, nhưng cũng rất thú vị và bổ ích.


## Kết luận

Trong bài học này, chúng ta đã tìm hiểu các ứng dụng thực tế của Xử lý ngôn ngữ tự nhiên (NLP). Chúng ta đã khám phá cách NLP vận hành các công nghệ như chatbot, phân tích cảm xúc và dịch ngôn ngữ. Việc hiểu các ứng dụng này giúp chúng ta nhận thấy tác động của NLP đến cuộc sống hàng ngày và tiềm năng giải quyết các vấn đề phức tạp.


## Tài liệu tham khảo

+ Sowmya Vajjala, Bodhisattwa Majumder, Anuj Gupta, và Harshit Surana, Practical natural language processing: a comprehensive guide to building real-world NLP systems. Sebastopol, Ca O’reilly Media, 2020.
+ IBM, “What is Natural Language Processing?,” IBM, 2023. https://www.ibm.com/topics/natural-language-processing
+ Coursera Staff, “What is Natural Language Processing? Definition and Examples,” Coursera, 16/06/2023. https://www.coursera.org/articles/natural-language-processing









