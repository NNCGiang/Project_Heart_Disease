import os
import pandas as pd # Thêm import pandas để đảm bảo DataFrame hoạt động

# Đường dẫn ban đầu gây lỗi FileNotFoundError vì tệp không tồn tại.
# Thay vào đó, chúng ta sẽ tải tệp lên.
def load_data():
    # Đường dẫn tới file CSV trong thư mục data
    file_path = os.path.join("data", "heart.csv")

    try:
        #Đọc dữ liệu từ file heart.csv
        df = pd.read_csv(file_path)
        print("Dữ liệu đã được tải thành công!")
        print(f"Kích thước dữ liệu: {df.shape}")
        print(f"Số hàng: {df.shape[0]}, Số cột: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {file_path}. Vui lòng kiểm tra lại.")
        return None
