---
layout: default
title: 1. Kỹ thuật viết code chuẩn
nav_order: 1
parent: tips
grand_parent: Coding Python
permalink: /python/tips/1_write_code/
---

# 📝 **Chương trình buổi học**

- **Phần 1:** Cơ bản về Clean Code và PEP-8
- **Phần 2:** Viết Code Pythonic
- **Phần 3:** Nguyên lý chung để viết code tốt
- **Phần 4:** Nguyên tắc SOLID và Design Patterns (Nâng Cao)

---

## 🧼 **Phần 1: Cơ bản về Clean Code và PEP-8**

### **1. Clean Code là gì?**

- Clean code là mã nguồn rõ ràng, dễ đọc, dễ bảo trì, có khả năng mở rộng và dễ dàng kiểm thử.
- Nó được viết theo cách mà người khác (đồng nghiệp) hoặc chính bạn trong tương lai có thể dễ dàng hiểu, cải tiến hoặc bảo trì.

![pic1.png](attachment:f8bab760-3274-4c12-9f6f-064726b7f7b0:pic1.png)

### **2. Nguyên tắc cốt lõi của Clean Code (3D1C)**

- **Dễ đọc và dễ hiểu (Readable):** Code nên được viết sao cho người khác có thể dễ dàng hiểu được mục đích và cách thức hoạt động.
- **Dễ bảo trì (Maintainable):** Clean code nên dễ dàng sửa đổi, mở rộng và tái sử dụng mà không gây ra lỗi phụ.
- **Dễ kiểm thử (Testable):** Code sạch nên được thiết kế để dễ dàng viết unit test và thực hiện kiểm thử tự động.
- **Có khả năng mở rộng (Extensible):** Code được thiết kế để dễ dàng thêm tính năng mới mà không cần thay đổi code hiện tại.

### **3. Tầm quan trọng của Clean Code**

- Clean Code là yếu tố quan trọng giúp tăng hiệu suất làm việc, cải thiện hợp tác nhóm và giảm chi phí bảo trì dài hạn.
    - **Giảm thiểu lỗi (Bugs):** Dễ dàng phát hiện và ngăn chặn lỗi tiềm ẩn, giảm thời gian debug và sửa lỗi.
    - **Tiết kiệm thời gian dài hạn:** Mặc dù ban đầu có thể tốn thời gian hơn, clean code giúp tiết kiệm rất nhiều thời gian trong quá trình bảo trì và phát triển sau này.
    - **Cải thiện làm việc nhóm:** Giúp các thành viên trong nhóm dễ dàng hiểu và làm việc với code của nhau, tăng hiệu quả hợp tác.
    - **Dễ dàng mở rộng:** Cấu trúc rõ ràng, dễ dàng thêm tính năng mới mà không làm ảnh hưởng đến hệ thống hiện tại.

### **4. Case study: Đặt tên biến, formatting**

- Clean code sử dụng tên biến/hàm có ý nghĩa, thêm tài liệu mô tả và tuân thủ các quy ước định dạng để tăng khả năng đọc hiểu.
- Ví dụ:

![pic2.png](attachment:a9c361f7-d445-482b-9e21-7e223fa65ab1:pic2.png)

### **5. Case study: Code dễ mở rộng**

- Code dễ mở rộng cho phép thêm chức năng mới mà không cần sửa đổi code hiện có. Thiết kế này sử dụng các hàm riêng biệt và cấu trúc dữ liệu như dictionary để quản lý các phép toán, giúp tuân thủ nguyên tắc "Open-Closed Principle".
- Ví dụ:

![pic3.png](attachment:e25366ce-3606-4c3e-9e3a-10e4950c0d83:pic3.png)

1. **Khó mở rộng (Bad example):** Sử dụng nhiều `elif` trong một hàm `calculate` cho các phép toán.
2. **Dễ mở rộng (Good example):** Tách mỗi phép toán thành một hàm riêng (`add`, `subtract`, `multiply`) và lưu chúng vào một dictionary `operations`. Hàm `calculate` sẽ gọi hàm tương ứng từ dictionary.

### **6. Khi nào có thể "bỏ qua" Clean Code?**

![pic4.png](attachment:7b97b49d-d857-4f5d-bd52-334226e2de76:pic4.png)

### **7. Giới thiệu về PEP-8**

- PEP-8 (Python Enhancement Proposal 8) là tiêu chuẩn định dạng code Python chính thức, quy định cách trình bày code để đảm bảo tính nhất quán và dễ đọc trên toàn dự án. Được tạo ra bởi Guido van Rossum - cha đẻ của Python.
- Ví dụ:

![pic5.png](attachment:e62dd2e9-6aaf-48d9-8813-108724359867:pic5.png)

### **8. Quy tắc đặt tên (Naming Convention)**

- **Biến/Hàm:** `snake_case` (chữ thường và dấu gạch dưới để phân tách các từ).
    
    ```python
    # Khai báo biến
    my_variable = 10
    
    # Định nghĩa hàm
    def my_function():
    		print("Hello")
    ```
    
- **Class:** `PascalCase` (Viết hoa chữ cái đầu của mỗi từ, không có dấu gạch dưới).
    
    ```python
    # Đặt tên class
    class MyClass:
    		def __init__(self, name):
    				self.name = name
    ```
    
- **Hằng số:** `UPPER_CASE` (Sử dụng tất cả chữ hoa và dấu gạch dưới giữa các từ).
    
    ```python
    # Đặt tên hằng số
    PI = 3.14
    MAX_VALUE = 100
    MIN_VALUE = 5
    
    def check_exam_result(score):
    		if score >= PASSING_SCORE:
    				return "Pass"
    		return "Fail"
    ```
    

### **9. Căn lề và khoảng trắng**

- **Thụt lề:** Dùng 4 khoảng trắng cho mỗi cấp thụt lề. Không dùng tab! (tức là cài setting trong IDE là tab = 4 spaces)
- **Khoảng trắng trong các biểu thức:** Thêm khoảng trắng hai bên phép toán: `a = b + c`. (ko phải là `a=b+c` )
- **Tránh khoảng trắng thừa:** Sau dấu phẩy: `f(a, b)` không phải `f(a , b)`.
- **Không thêm khoảng trắng trong ngoặc:** Đúng: `list([1, 2, 3])` Sai: `list( [1, 2, 3] )`.

### **10. Giới hạn độ dài dòng code**

![pic6.png](attachment:62b6c9bb-40e7-43a2-a22c-10dd6312e542:pic6.png)

Việc giới hạn giúp ta chia màn hình code → đọc được nhiều file nhưng vẫn giữ đủ nội dung trong từng dòng code để đọc → tránh quá nhỏ khiến lướt nhiều hoặc quá dài thì ko đủ split màn hình

### **11. Tổ chức Import đúng chuẩn**

1. **Thư viện chuẩn:** Các module từ thư viện chuẩn Python như `os`, `sys`, `math...`
2. **Thư viện bên thứ ba:** Các package được cài qua pip như `requests`, `numpy`, `pandas...`
3. **Module nội bộ:** Các module tự viết trong dự án của bạn.

![pic7.png](attachment:9239faa2-a81a-4ff7-bbd7-d37817293da6:pic7.png)

### **12. Dòng trắng và cấu trúc Hàm/Class**

- **Dòng trắng giữa các hàm và class:** Sử dụng 2 dòng trắng để phân tách các định nghĩa class. Các hàm bên trong class cách nhau 1 dòng trắng.
    
    ```python
    class Student():
    		self.name = ""
    		self.age = 10
    		
    		def __init__(self):
    				self.name = "NULL"
    				self.age = 0
    		
    		def print_info(self):
    				print(self.name + " - " + self.age)
    	
    	
    class Teacher():
    		#code	
    ```
    
- **Dòng trắng trong hàm:** Sử dụng dòng trắng để phân tách các nhóm logic code. Không nên quá nhiều dòng trắng gây khó đọc.
    
    ```python
    def cal_sum_lst(lst):
    		sum_lst = 0
    		for i in lst:
    				sum_lst += i
    				
    		return sum_lst
    ```
    
- **Cấu trúc class chuẩn:** `docstring` → biến class → `__init__()` → các phương thức khác → các phương thức private (method) → các static method.
    
    ```python
    class Student():
    		"""
    		define info of student
    		
    		"""	
    		self.name = ""
    		self.age = 10
    		
    		def __init__(self):
    				self.name = "NULL"
    				self.age = 0
    		
    		def print_info(self):
    				print(self.name + " - " + self.age)
    ```
    
- **Cấu trúc hàm:** `docstring` → các lệnh validate đầu vào → code chính → `return`. Mỗi hàm nên có một mục đích duy nhất.
    
    ```python
    def cal_sum_lst(lst):
        """calculate sum of list
    
        Args:
            lst (List): input list of numbers
    
        Returns:
            int: sum of numbers in the list
        """
        sum_lst = 0
        for i in lst:
                sum_lst += i
                
        return sum_lst
    ```
    

### **13. Tài liệu hóa (Documentation)**

- Tài liệu hóa code giúp nâng cao khả năng bảo trì, giảm thời gian đọc hiểu, và tạo điều kiện cho việc hợp tác trong đội ngũ phát triển.
- **Viết docstring hiệu quả:**
    - Đặt trong cặp dấu ba nháy kép (`"""`) ngay sau định nghĩa hàm.
    - Các định dạng phổ biến: Google style (dễ đọc), NumPy style (chi tiết), reStructuredText (tích hợp Sphinx).
- **Annotations và type hints (PEP 484):** Cung cấp kiểm tra kiểu dữ liệu tĩnh.
    - Kiểu cơ bản: các kiểu dữ liệu có sẵn như `int`, `float` hay `string`
        
        ```python
        def add(a: int, b: int) -> float:
        	#code
        ```
        
    - Kiểu phức tạp: `List`, `Dict`, `Tuple`, `Optional`, `Union`.
        
        ```python
        from typing import List, Dict, Tuple, Optional 
        #(Optional for type of result we unsure)
        
        def  add_number_to_list(lst: List, n: int) -> List:
        		#code
        ```
        
        > *Annotations và Hint không ảnh hưởng đến cú pháp hay logic code, nó chỉ đơn giản là cách ta tự kiểm soát đầu vào và đầu ra của hàm (do Python khác với các ngôn ngữ khác, không yêu cần kiểu dữ liệu đầu vào và đầu ra)*
        > 
    - Công cụ kiểm tra: `mypy`, Pyright, PyCharm.
- **Tài liệu module và project:**
    - `Module docstring`: Đặt ở đầu file, mô tả mục đích và các hàm chính.
    - `README.md`: Hướng dẫn cài đặt, ví dụ đơn giản, cấu trúc project.
    - `CONTRIBUTING.md`: Quy trình đóng góp, quy tắc code.
    - Sphinx: Tự động tạo tài liệu API từ docstring.
    - Wiki: Tài liệu chi tiết cho các tính năng phức tạp.

### **14. Công cụ kiểm tra PEP-8 và code cơ bản**

![pic8.png](attachment:b16180bb-a58a-4d44-ad20-c6bfa4f611fa:d6ea5660-a255-48c8-a503-00af54f85dd8.png)

!!! Bài tập thực hành

```python
def calculate_BMI(height: int, weight: int) -> str:
    """calculate BMI score base on height and weight

    Args:
        height (int): height in cm
        weight (int): weight in kg

    Returns:
        str: BMI status
    """
    BMI = weight / (height ** 2)
    if BMI <= 18.5:
        return "Thiếu cân"
    elif BMI <= 25:
        return "Bình thường"
    else:
        return "Thừa cân"
    
print("\nResult:")
print(calculate_BMI(height=170, weight=70))
```

### **15. Cấu trúc tiêu chuẩn cho dự án Python và Data Science (Tham khảo)**

- **Dự án Python chuẩn:** Thư mục gốc (chứa README, requirements.txt, setup.py), Mã nguồn (src/), Tests (tests/), Tài liệu (docs/), Tài nguyên (resources/).
- **Dự án Data Science chuẩn (Cookiecutter Data Science):** Bao gồm các thư mục `data` (raw, interim, processed, external), `docs`, `models`, `notebooks`, `references`, `reports` (figures), và `src` (data, features, models, visualization).

---

## 🐍 **Phần 2: Viết Code Pythonic**

### **1. The Zen of Python (Triết lý Python)**

- Tập hợp 19 nguyên tắc hướng dẫn thiết kế Python, được Tim Peters viết trong PEP 20. Có thể xem bằng cách gõ `import this` trong Python.
- **Một số nguyên tắc tiêu biểu:**
    - Beautiful is better than ugly (Cái đẹp tốt hơn cái xấu).
    - Simple is better than complex (Đơn giản tốt hơn phức tạp).
    - Explicit is better than implicit (Rõ ràng tốt hơn ẩn ý).

### **2. Pythonic Code là gì?**

- Pythonic nghĩa là tận dụng tối đa tính năng và đặc điểm riêng của Python để viết code.
- Code Pythonic dễ đọc, dễ hiểu, ngắn gọn như đọc tiếng Anh, đồng thời tuân thủ các quy ước và triết lý của Python.

![pic9.png](attachment:5c24bd8a-0580-4ffa-ae7f-4cbb53f00ef8:pic9.png)

### **3. Indexes và Slices**

- Python cung cấp cách truy cập mạnh mẽ vào các phần tử trong sequences (list, tuple, string) thông qua indexes và slicing.
- Cú pháp slicing `sequence[start:stop:step]` giúp thao tác với dữ liệu linh hoạt và Pythonic.

![pic10.png](attachment:1bb4e629-b6da-43a9-aa20-24f8a37a2eac:pic10.png)

### **4. List, Dict, Set Comprehensions**

- Giúp code gọn và nhanh hơn.

![pic11.png](attachment:86e67fd1-61ac-4925-bc02-db85c85cc0fc:pic11.png)

### **5. Context Managers (with)**

- Context Manager là cơ chế quản lý tài nguyên thông qua câu lệnh `with`, giúp tự động giải phóng tài nguyên (đóng file, đóng kết nối DB, giải phóng lock...) khi khối lệnh kết thúc, kể cả khi có lỗi xảy ra.

![pic12.png](attachment:28b6f678-a8ac-4cfb-acb5-6b7bf4f402c1:pic12.png)

- **Ưu điểm:** Đảm bảo tài nguyên luôn được giải phóng đúng cách, tránh memory leak, giúp code ngắn gọn và an toàn hơn khi xử lý ngoại lệ.
- **Ứng dụng phổ biến:** Quản lý file, kết nối database, lock thread, benchmark thời gian, tạm thay đổi cấu hình, transaction database.

### **6. So sánh và Điều kiện - Pythonic Style**

- **Trường hợp 1: So sánh với None**
    
    ![pic13.png](attachment:fd311e50-3c99-4801-83b1-1893de7420c2:pic13.png)
    
    - **Giải thích:** `None` là singleton object, phải dùng `is`/`is not` để so sánh về mặt identity.
- **Trường hợp 2: So sánh Boolean**
    
    ![pic14.png](attachment:daa2f7e5-bd0a-47d5-b715-a2d2275f54fa:pic14.png)
    
    - **Giải thích:** Boolean tự thân đã có giá trị truth, không cần so sánh thêm.
- **Trường hợp 3: Kiểm tra chuỗi/list/dict rỗng**
    
    ![pic15.png](attachment:04c98914-88c2-413a-b752-9985e98cf0c4:pic15.png)
    
    - **Giải thích:** Chuỗi/list/dict rỗng được đánh giá là `False`, không rỗng là `True` trong ngữ cảnh boolean.
- **Trường hợp 4: Kiểm tra trong collection**
    
    ![pic16.png](attachment:2e1163aa-1034-4a6e-9c1c-137e14167d9b:pic16.png)
    
    - **Giải thích:** Sử dụng toán tử `in` trực tiếp, hiệu quả và dễ đọc hơn.
- **Trường hợp 5: Chaining comparison (So sánh chuỗi)**
    
    ![pic17.png](attachment:a2aca88e-694f-4fdb-a1d1-8eaf4fda1c89:pic17.png)
    
    - **Giải thích:** Python cho phép "chain" các phép so sánh, giống như toán học, giúp dễ đọc và hiểu hơn.

### **7. Properties và dấu underscore**

- Sử dụng `@property` cho phép sử dụng method như thuộc tính, giúp code gọn, dễ kiểm soát truy cập. Tuân theo quy chuẩn sử dụng dấu underscore giúp tránh lỗi khi thiết kế class.

![pic18.png](attachment:8f3c6fb7-971f-4670-8d6c-964a7098dacd:pic18.png)

- **Dấu underscore:**
    - **`_var` (Single Underscore):** Quy ước cho biến private hoặc "internal use" trong module, class. Không thực sự ngăn truy cập từ bên ngoài, nhưng là dấu hiệu "không nên sử dụng trực tiếp".
    - **`__var__` (Double Underscore hai bên):** Dành cho phương thức đặc biệt (magic/dunder methods) như `__init__`, `__str__`. Không nên tự tạo phương thức kiểu này trừ khi cần thiết.
    - **`__var` (Double Underscore):** Name mangling - Python tự động đổi tên thành `_ClassName__var` để tránh xung đột khi kế thừa. Giúp thuộc tính không bị ghi đè bởi lớp con một cách vô tình.
    - **`_` (Một dấu gạch dưới):** Thường dùng làm biến tạm không quan trọng hoặc kết quả gần nhất trong Python interpreter.

!!! Bài tập thực hành

```python
# Ví dụ 1
nums = [1, 2, 3, 4, 5]

result = [num ** 2 for num in nums if num % 2 == 0] #[4, 16]

# Ví dụ 2
names = ['Anna', 'Bob', 'Charlie']
ages = [25, 30, 35]

info = {name: age for name, age in zip(names, ages)} # {'Anna': 25, 'Bob': 30, 'Charlie': 35} 

# Ví dụ 3
numbers = [10, 5, 8, 20, 3]
max_num = max(numbers) # 20
```

---

## ✨ **Phần 3: Nguyên lý chung để viết code tốt**

### **1. Tổng quan các nguyên lý chung**

- Nguyên lý chung để viết code tốt hơn dựa trên các nền tảng cốt lõi như DRY, YAGNI, KISS, lập trình phòng thủ và phân chia trách nhiệm.
- Những nguyên lý này giúp tạo code bền vững và dễ bảo trì.

### **2. DRY (Don't Repeat Yourself)**

- **Nguyên tắc:** "Tránh lặp lại code, mỗi phần kiến thức trong hệ thống phải có một biểu diễn duy nhất, rõ ràng, và có thẩm quyền”.

![pic19.png](attachment:b4b6636c-2605-4b5d-ac43-8835c4b9e124:pic19.png)

- **Lợi ích:** Giảm lỗi khi cần thay đổi logic, code ngắn gọn, dễ bảo trì hơn, tăng khả năng tái sử dụng code.
- **Cách áp dụng:** Tách các đoạn code lặp lại thành functions, classes, hoặc modules riêng để tái sử dụng.

### **3. YAGNI (You Aren't Gonna Need It)**

- **Nguyên tắc:** Khuyên lập trình viên không nên thêm chức năng cho đến khi thực sự cần thiết.

![pic1.png](attachment:a982149e-214a-40e7-917b-16b26c3336ef:eb963280-7a74-493a-871c-31a724992f07.png)

- **Lợi ích:** Tránh lãng phí thời gian phát triển các tính năng không cần thiết, giảm thiểu technical debt, giữ code đơn giản, dễ bảo trì hơn.
- **Khi nào áp dụng:** Khi bạn đang muốn thêm một tính năng "phòng khi cần" mà chưa có yêu cầu cụ thể.

### **4. KISS (Keep It Simple, Stupid)**

- **Nguyên tắc:** Ưu tiên giải pháp đơn giản và dễ hiểu nhất. Tránh các giải pháp phức tạp khi không cần thiết.

![pic2.png](attachment:eec4b9fe-281f-43dc-a10b-b82f3c1e1b3a:pic2.png)

- **Lợi ích:** Code đơn giản dễ dàng đọc, debug, bảo trì và mở rộng. Giảm thiểu bug và tăng hiệu suất làm việc nhóm.
- **Khi nào áp dụng:** Tận dụng triết lý "Đơn giản hơn tốt hơn phức tạp" của Python để viết code rõ ràng, ngắn gọn.

### **5. Defensive Programming (Lập trình phòng thủ)**

- **Nguyên tắc:** Kỹ thuật viết code luôn giả định rằng sẽ có lỗi xảy ra. Bao gồm kiểm tra đầu vào, xử lý ngoại lệ và kiểm tra điều kiện biên để tránh lỗi không mong muốn.

![pic3.png](attachment:ac392987-34a9-42d7-8b7f-fd51f6cb7a44:pic3.png)

- **Nguyên tắc cơ bản:** Luôn kiểm tra đầu vào, xác thực dữ liệu từ người dùng, file, API và các nguồn bên ngoài. Không bao giờ tin tưởng input từ bất kỳ nguồn nào.
- **Lợi ích:** Tạo ra code bền vững, có khả năng ứng phó với các tình huống không lường trước và dễ dàng debug khi có vấn đề.
- **Kết hợp:** Trong Python, hãy kết hợp kiểm tra kiểu dữ liệu, `assert`, và `try-except` để xây dựng code an toàn.

### **6. Xử lý lỗi (Error handling)**

- Luôn bắt lỗi cụ thể thay vì chung chung (`Exception`). Không bao giờ "bỏ qua lỗi" mà không xử lý hoặc ghi log.

![pic4.png](attachment:4c456229-329c-4156-8317-58537bbd7bb3:pic4.png)

- **Xử lý lỗi tốt:** Không chỉ là bắt lỗi mà còn là truyền thông tin lỗi đúng cách. Sử dụng `logging` thay vì `print` và tạo custom exceptions khi cần thiết để làm rõ ngữ cảnh lỗi.

### **7. Phân chia trách nhiệm (Separation of Concerns)**

- **Nguyên tắc:** Phân chia code thành các module, class hoặc hàm riêng biệt, mỗi phần chỉ đảm nhiệm một chức năng cụ thể. Giúp dễ bảo trì, dễ kiểm thử và tăng khả năng tái sử dụng.

![pic5.png](attachment:ea63d6a9-572f-44c5-8486-b41b968a8a33:pic5.png)

### **8. Sử dụng Logging và Print hợp lý**

- **Print:** Đơn giản, dễ sử dụng nhưng khó kiểm soát đầu ra, không phân loại mức độ nghiêm trọng, khó tắt khi triển khai, không lưu lại thông tin như thời gian.
- **Logging:** Cung cấp tính năng ghi nhật ký chuyên nghiệp với nhiều cấp độ và tùy chọn cấu hình. Cấu hình linh hoạt, nhiều cấp độ log (DEBUG, INFO, WARNING...), tự động thêm thời gian, file, dòng code, có thể điều hướng log đến file, email....

!!! Bài tập thực hành

```python
def greeting(name, language):
    dictionary = {
        'vn': "Xin chào",
        'en': "Hello",
        'fr': "Bonjour"
    }
    if language not in dictionary:
        raise ValueError ("Không tồn tại ngôn ngữ đầu vào")
    
    return f"{dictionary[language]}, {name}"
```

---

## 🏗️ **Phần 4: Nguyên tắc SOLID và Design Patterns (Nâng Cao)**

### **1. Giới thiệu SOLID Principles**

- **S - Single Responsibility Principle:** Mỗi lớp chỉ nên có một lý do để thay đổi. Tập trung vào một nhiệm vụ duy nhất.
- **O - Open/Closed Principle:** Mở để mở rộng, đóng để sửa đổi. Code nên dễ mở rộng mà không cần sửa đổi.
- **L - Liskov Substitution Principle:** Các lớp con phải thay thế được lớp cha. Đảm bảo tính nhất quán.
- **I - Interface Segregation Principle:** Nhiều interface nhỏ tốt hơn một interface lớn. Tránh phụ thuộc không cần thiết.
- **D - Dependency Inversion Principle:** Phụ thuộc vào abstraction, không phụ thuộc vào cụ thể. Giảm sự ràng buộc.

### **2. Áp dụng SOLID vào Python**

- **Tính linh hoạt (Dynamic Typing):** Python dynamic typing cho phép dễ dàng thay đổi hành vi object trong runtime. Giúp thiết kế các interface linh hoạt, hữu ích cho Open/Closed Principle.
- **Duck Typing:** Các objects chỉ cần hiện thực các phương thức tương thích, không cần kế thừa, hỗ trợ tốt cho Liskov Substitution và Interface Segregation.
- **Abstractions (Module `abc`):** Cung cấp `@abstractmethod` decorator và `ABC` class để định nghĩa interface thuần khiết. Đảm bảo các lớp con phải triển khai đúng các phương thức được yêu cầu, tăng cường Dependency Inversion.
- **Composition:** Python khuyến khích "composition over inheritance" thông qua mixins và dependency injection. Việc kết hợp các đối tượng nhỏ giúp đạt Single Responsibility và giảm phụ thuộc giữa các module.

### **3. Single Responsibility Principle (SRP)**

- **Một Lớp, Một Nhiệm Vụ:** Mỗi class chỉ nên có một lý do để thay đổi. Ví dụ: `DatabaseConnector` chỉ kết nối database, không xử lý logic nghiệp vụ.
- **Phân Tách Rõ Ràng:** Chia nhỏ các lớp lớn thành các module nhỏ hơn. Ví dụ: `FileProcessor` thành `FileReader`, `FileValidator`, `FileParser`.
- **Dễ Bảo Trì Và Test:** Khi mỗi class chỉ làm một việc, code dễ hiểu và dễ sửa. Các đơn vị nhỏ cũng dễ kiểm thử hơn.
- **Ví dụ:**

![pic6.png](attachment:53d70a2e-cd3e-4302-a3d7-18a45129bb62:pic6.png)

### **4. Open/Closed Principle (OCP)**

- **Mở Cho Việc Mở Rộng:** Dễ dàng thêm chức năng mới mà không làm thay đổi code hiện có.
- **Đóng Cho Việc Sửa Đổi:** Tránh sửa đổi nội dung bên trong class đã được kiểm thử và triển khai. Thay vào đó, mở rộng thông qua kế thừa hoặc composition.
- **Sử Dụng Abstraction:** Tạo các abstract base class (ABC) hoặc protocol để định nghĩa interface.
- **Mở Rộng Bằng Kế Thừa:** Tạo subclass `Triangle`, `Square` kế thừa từ `Shape` thay vì sửa đổi `Shape`. Strategy pattern và Plugin architecture là hai kỹ thuật hiệu quả.

### **5. Liskov Substitution Principle (LSP)**

- **Thay Thế Được:** Đối tượng của lớp con phải hoạt động đúng khi được sử dụng thay cho đối tượng lớp cha.
- **Hành Vi Nhất Quán:** Lớp con không được thay đổi hành vi được định nghĩa trong interface/lớp cha. Ví dụ: `Penguin` không nên triển khai `fly()` bằng cách ném ra exception nếu `Bird` có `fly()`.
- **Tránh Vi Phạm:** Không thêm precondition mạnh hơn, postcondition yếu hơn, hoặc ném ra exception mới trong lớp con.

### **6. Interface Segregation Principle (ISP)**

- **Tách Nhỏ Interface:** Nhiều interface chuyên biệt tốt hơn một interface lớn. Ví dụ: tách `PrinterInterface` thành `Scanner`, `Printer`, `Fax` thay vì `AllInOnePrinter`.
- **Tránh "Fat Interface":** Client không nên bị buộc triển khai phương thức không cần.
- **Trong Python:** Dùng `ABC` module hoặc `typing.Protocol` để định nghĩa interface rõ ràng. Duck typing cho phép các lớp chỉ triển khai những phương thức cần thiết.
- **Lợi Ích:** Code linh hoạt hơn, dễ unit test, giảm coupling và side-effects.

### **7. Dependency Inversion Principle (DIP)**

- **Nguyên tắc:** Các module cấp cao không nên phụ thuộc trực tiếp vào module cấp thấp, mà cả hai nên phụ thuộc vào abstraction.
- **High-level Module:** Định nghĩa logic nghiệp vụ, không nên phụ thuộc trực tiếp vào chi tiết triển khai cụ thể.
- **Abstraction:** Cả module cấp cao và cấp thấp đều phụ thuộc vào abstraction (interfaces/protocols).
- **Low-level Module:** Triển khai các abstractions được định nghĩa bởi module cấp cao.

### **8. Các Design Patterns Phổ Biến**

- **Creational Patterns (Mẫu khởi tạo):** Tạo đối tượng theo cách linh hoạt. `Factory` và `Singleton` là phổ biến nhất.
    - **Factory Pattern:** Tạo đối tượng mà không cần biết lớp con cụ thể. Ẩn logic phức tạp, cung cấp interface chung, dễ mở rộng.
    - **Singleton Pattern:** Chỉ tạo duy nhất một instance, thường dùng cho logging, database. Truy cập toàn cục, kiểm soát tài nguyên chia sẻ, tiết kiệm bộ nhớ.
- **Structural Patterns (Mẫu cấu trúc):** Xác định mối quan hệ giữa các đối tượng. `Adapter` và `Decorator` rất hữu ích.
    - **Adapter Pattern:** Kết nối interface không tương thích. Trung gian phiên dịch, tích hợp thư viện bên thứ ba, xử lý khác biệt API, không sửa code gốc.
    - **Decorator Pattern:** Thêm chức năng mới cho đối tượng bằng cách bao bọc bằng Wrapper. Mở rộng chức năng linh hoạt, tuân thủ OCP, thêm nhiều tính năng kết hợp.
- **Behavioral Patterns (Mẫu hành vi):** Xác định cách giao tiếp giữa các đối tượng. `Command` và `Template Method` thường dùng.
    - **Command Pattern:** Đóng gói yêu cầu thành đối tượng độc lập. Tách biệt Invoker và Receiver, hỗ trợ lưu lệnh, nhật ký hóa thao tác, giao dịch phân tán. Dễ mở rộng lệnh mới, tuân thủ OCP. Ứng dụng: GUI (nút), giao dịch tài chính, undo/redo.
    - **Template Method Pattern:** Định nghĩa khung thuật toán cố định. Lớp con tùy chỉnh các bước cụ thể, tái sử dụng code. Hook methods để kiểm soát các bước trong thuật toán. Thường dùng cho thuật toán xử lý dữ liệu tuần tự: đọc, xử lý, ghi.

![pic7.png](attachment:db4195bf-57f4-41e8-b4e8-742cb8670236:pic7.png)

### **9. Cách Dùng Design Patterns Hiệu Quả**

![pic8.png](attachment:cf87748f-b079-40b4-830f-08285e22dee4:pic8.png)

---

## 🎯 **Tổng kết buổi học**

- **Phần 1: Clean Code & Formatting:** Tiêu chuẩn PEP-8, tài liệu hóa code với docstring và type annotations, công cụ kiểm tra code và tích hợp CI.
- **Phần 2: Pythonic Code:** List/Dict comprehensions, context managers, properties, assignment expressions và các kỹ thuật lập trình đặc trưng của Python.
- **Phần 3: Nguyên lý code sạch:** DRY, YAGNI, KISS, defensive programming, design by contract và separation of concerns.
- **Phần 4: SOLID & Design Patterns:** 5 nguyên tắc SOLID và các design patterns phổ biến: Factory, Singleton, Adapter, Decorator, Command.

---

## 📚 **Tài liệu tham khảo**

![pic9.png](attachment:1e209e55-233a-47ef-9fd1-80cc2cd30728:pic9.png)

1. Clean Code in Python, Mariano Anaya.
2. Effective-Python, Brett Slatkin.
3. The Art Of Readable Code, Dustin Boswell and Trevor Foucher.