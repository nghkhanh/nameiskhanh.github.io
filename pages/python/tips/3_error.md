---
layout: default
title: 3. Các lỗi thường gặp
nav_order: 3
parent: Tips
grand_parent: Coding Python
permalink: /python/tips/3_error/
---

## Common Erros

### NameError

Xảy ra khi một tên biến hoặc tên hàm, thư viện không được định nghĩa trong ngữ cảnh chương trình, xuất hiện khi tên hàm, tên biến không tồn tại, hoặc đã bị xóa, hoặc không tồn tại trong phạm vị hàm hoặc chương trình. 

```python
a = 4
c = a + b #NameError do b chưa được định nghĩa
print(c)

#---------------------------------------------------------#

a = 5
Print(a) #NameError do Python phân biệt chữ hoa và thường, nên chỉ có hàm print()

#---------------------------------------------------------#

def a_function(x):
		a_variable = 4
		result = x * a_variable
		
		return result
		
print(a_vaiable) #NameError do a_variable được khai báo trong hàm, nên khi gọi ngoài hàm sẽ không tồn tại
```

### SyntaxError

Xảy ra khi phát hiện cấu trúc code ta viết không hợp lệ với Python. Ví dụ như thiếu/thừa ngoặc đơn, dùng sau từ khóa, hay thiếu dấu ‘:’ sau if → Nói chung là các lỗi liên qua đến ‘chính tả’ trong code Python.

```python
s = 'Hello World" #SyntaxError do chuỗi có dấu đơn ở đầu và dấu đôi ở cuối → cần thống nhất dấu ở đầu và đuôi

print(s)

#---------------------------------------------------------#

import math

number = 20.2
print(math.floor(number) #SyntaxError do thiếu dấu ngoặc đơn đóng ở cuối

#---------------------------------------------------------#

print "Hello Word" #SyntaxError do cặp dấu ngoặc đơn

#---------------------------------------------------------#

number = 10
if number > 15         #SyntaxError do thiếu dấu ‘:’ sau câu lệnh if
		print("Large number")

```

### ZeroDivisionError

Xảy ra khi ta thực hiện phép chia với mẫu bằng 0 

```python
a = 10
b = 0

c = a / b #ZeroDivisionError do chia với giá trị 0 (b = 0)

print(c) 
```

### TypeError

Xảy ra khi thực hiện các phép toán hoặc hàm với dữ liệu có kiểu không phù hợp. VD: cộng chuỗi và số 

```python
a = 'AI'

b = 5

c = a + b #TypeError, do trong Python không cho phép cộng chuỗi và số

print(c)
```

### IdentationError

Xảy ra khi việc thụt lề (identation) không nhất quán (có nơi thụt 2 space, nơi thì 3 space, …) 

```python
a = 5
	b = 6
	
c = a + b #IdentationError do b thụt vào nhưng không có nằm trong khối mã nào

print(c)
```

### ModuleNotFoundError

Xảy ra khi Python không tìm thấy module (thư viện) ta cố import vào thư viện

> *Module (thư viện) là tập hợp các đoạn code được đóng gói lại để sử dụng trong các chương trình khác nhau*
> 

Một số nguyên nhân:

- Module chưa được cài đặt: sử dụng lệnh pip install <module_name>
- Đường dẫn file chưa chính xác: khi module của ta là một file thì kiểm tra lại đường dẫn
- PYTHONPATH (ít xảy ra): khi PYTHONPATH không được cài đặt đúng

```python
import mymodule  #ModuleNotFoundError do mymodule chưa được cài hoặc không tìm thấy trong workspace hiện tại

print("Hello World")
```

### IndexError

Xảy ra khi ta truy cập vào chỉ mục không tồn tại trong list hoặc tuple. 

!!!Lưu ý, Python sử dụng chỉ mục từ 0 đến n - 1, với n là số lượng phần tử trong list/tuple

```python
lst = [1, 2, 3, 4, 5]

print(lst[5])  #IndexError do index 5 không tồn tại
print(lst[0])
```

### ValueError

Xảy ra khi hàm hoặc toán tử chấp nhận kiểu dữ liệu, nhưng giá trị lại không được chấp nhận với hàm đó (cụ thể tùy hàm)

```python
import math

number = -4
print(math.sqrt(number)) #ValueError do không thể tính được căn bậc hai của số âm

#---------------------------------------------------------#

my_string = "Day la bai hoc dau tien"

my_string.index("Hello") #ValueError do giá trị ta tìm (”Hello”) không có trong chuỗi, 
													#có thể tránh bằng cách kiểm tra giá trị đó có tồn tại trước khi dùng index()
													

#---------------------------------------------------------#

str1 = '5'
str2 = 'hello'

value1 = int(str1) #hợp lệ cho chuỗi ‘5’ có thể chuyển thành số nguyên
value2 = int(str2) #ValueError do ‘hello’ không phải giá trị hợp lệ để thành số nguyên
```

### RecursionError

Xảy ra khi hàm đệ quy không kết thúc hoặc quá nhiều dẫn đến vượt giới hạn đệ quy (recursion limit)

Mỗi lần hàm đệ quy được gọi, 1 frame mới được thêm vào stack của chương trình. Khi stack tràn, lỗi RecursionError xảy ra

```python
def a_function(n):
		return a_function(n)

a_function(5) #hàm sẽ chạy mãi mãi → Recursion xảy ra
							#đảm bảo khi sử dụng đệ quy phải cho điều kiện dừng hợp lý để chấm dứt quá trình đệ quy
```