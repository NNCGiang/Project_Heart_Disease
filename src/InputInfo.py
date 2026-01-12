import os
import pandas as pd
import joblib
from preprocessing import Preprocessor

MODEL_DIR = os.path.join(os.getcwd(), "models")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# ---- Load preprocessor ----
if not os.path.exists(preprocessor_path):
    raise FileNotFoundError(f"Không tìm thấy preprocessor.pkl. Hãy chạy main.py để tạo file này!")
pre = joblib.load(preprocessor_path)

model_files = {
    "MLP": "best_MLP.pkl",
    "DecisionTree": "best_DecisionTree.pkl",
    "GradientBoosting": "best_GradientBoosting.pkl"
}

# ================= NHẬP DỮ LIỆU =================
def get_float(prompt, mn, mx):
    while True:
        try:
            v = float(input(prompt))
            if mn <= v <= mx:
                return v
        except:
            pass
        print(f"Giá trị không hợp lệ! Vui lòng nhập từ {mn} đến {mx}.")

def get_choice(prompt, choices):
    while True:
        try:
            v = int(input(prompt))
            if v in choices:
                return v
        except:
            pass
        print(f"Giá trị không hợp lệ! Chọn trong {choices}.")

def input_patient_data():
    print("\n" + "=" * 60)
    print("NHẬP THÔNG TIN BỆNH NHÂN")
    print("=" * 60)
    patient_data = {}

    # Numeric
    patient_data["Age"] = get_float("Age [20-77]: ", 20, 77)
    patient_data["Sex"] = get_choice("Sex (0=Female, 1=Male): ", [0, 1])

    # ChestPainType mapping
    chest_map = {0:"TA", 1:"ATA", 2:"NAP", 3:"ASY"}
    cp_choice = get_choice("ChestPainType (0=TA,1=ATA,2=NAP,3=ASY): ", [0,1,2,3])
    patient_data["ChestPainType"] = chest_map[cp_choice]

    patient_data["RestingBP"] = get_float("RestingBP: ", 0, 200)
    patient_data["Cholesterol"] = get_float("Cholesterol: ", 0, 600)
    patient_data["FastingBS"] = get_choice("FastingBS (0-1): ", [0,1])

    # RestingECG mapping
    rest_map = {0:"Normal", 1:"ST", 2:"LVH"}
    rest_choice = get_choice("RestingECG (0=Normal,1=ST,2=LVH): ", [0,1,2])
    patient_data["RestingECG"] = rest_map[rest_choice]

    patient_data["MaxHR"] = get_float("MaxHR: ", 60, 202)
    patient_data["ExerciseAngina"] = get_choice("ExerciseAngina (0-1): ", [0,1])
    patient_data["Oldpeak"] = get_float("Oldpeak: ", -2.6, 6.2)

    # ST_Slope mapping
    slope_map = {0:"Up", 1:"Flat", 2:"Down"}
    slope_choice = get_choice("ST_Slope (0=Up,1=Flat,2=Down): ", [0,1,2])
    patient_data["ST_Slope"] = slope_map[slope_choice]

    return patient_data


# ================= CHỌN MODEL =================
def choose_model():
    print("\nChọn mô hình dự đoán:")
    for i, key in enumerate(model_files.keys(), start=1):
        print(f"{i}. {key}")
    while True:
        try:
            choice = int(input("Nhập số lựa chọn: "))
            if choice in range(1, len(model_files)+1):
                model_name = list(model_files.keys())[choice-1]
                model_path = os.path.join(MODEL_DIR, model_files[model_name])
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Không tìm thấy {model_files[model_name]} trong {MODEL_DIR}")
                model = joblib.load(model_path)
                return model_name, model
        except Exception as e:
            print(f"Lỗi: {e}")

# ================= DỰ ĐOÁN =================
def run_input_pipeline():
    model_name, model = choose_model()
    features = [
        "Age","Sex","ChestPainType","RestingBP","Cholesterol",
        "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"
    ]

    while True:
        data = input_patient_data()
        X_raw = pd.DataFrame([[data[f] for f in features]], columns=features)
        try:
            X = pre.transform_new_data(X_raw)
        except Exception as e:
            print(f"Lỗi tiền xử lý dữ liệu: {e}")
            continue
        if model_name == "MLP" and hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]

            # Load ngưỡng đã lưu (nếu có), mặc định 0.5
            threshold_path = os.path.join(MODEL_DIR, "best_threshold.pkl")
            if os.path.exists(threshold_path):
                threshold = joblib.load(threshold_path)
                # Nếu threshold > 1 thì chuẩn hóa về thang 0–1
                if threshold > 1:
                    threshold = threshold / 100.0
            else:
                threshold = 0.5

            y_pred = int(prob >= threshold)

        else:
            # Các mô hình khác dùng predict() mặc định
            y_pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        print("\n" + "="*40)
        print(f"Mô hình: {model_name}")
        print("KẾT QUẢ: CÓ BỆNH TIM" if y_pred==1 else "KẾT QUẢ: KHÔNG CÓ BỆNH TIM")
        if prob is not None:
            print(f"Xác suất: {prob:.2%}") #Xác xuất 1 người có khả năng mắc vấn đề về tim mạch

        if input("Tiếp tục dự đoán? (y/n): ").lower() != "y":
            break

# ================= MAIN =================
if __name__ == "__main__":
    run_input_pipeline()
