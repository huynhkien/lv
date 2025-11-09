import csv

# Đường dẫn để tạo file CSV
csv_file_path = 'dataset.csv'

# Tạo danh sách tên giả định hoặc có thể tự đặt tên theo ý của bạn
names = ['Tên_' + str(i) for i in range(1, 2601)]  # Ví dụ: ['Tên_1', 'Tên_2', ...]

# Đảm bảo danh sách `names` có độ dài 2600
if len(names) != 2600:
    print("Danh sách tên phải có độ dài bằng 2600!")
else:
    # Tạo file CSV với mã hóa UTF-8
    with open(csv_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Viết tiêu đề cột
        writer.writerow(['filename', 'name'])

        # Tạo dữ liệu từ 1.png đến 2600.png
        data = [[f"{i}.png", names[i-1]] for i in range(1, 2601)]

        # Ghi dữ liệu vào file CSV
        writer.writerows(data)

    print("Tạo file CSV thành công!")
