import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_processed = None
        self.label_encoders = {}   # khởi tạo rỗng để tránh lỗi attribute
        self.onehot_encoder = None
        self.scaler = None
        self.X = None
        self.y = None

    def clean_data(self):
        """
        Tiền xử lý dữ liệu an toàn:
        - Ép tất cả cột numeric sang float để tránh lỗi isnan
        - Thay giá trị 0 bằng NaN cho Cholesterol và RestingBP
        - Điền missing bằng KNNImputer
        """
        df_processed = self.df.copy()

        # Loại bỏ cột AgeGroup nếu có
        if 'AgeGroup' in df_processed.columns:
            df_processed = df_processed.drop('AgeGroup', axis=1)

        # ===== Xác định các cột số =====
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

        # Ép sang float
        for col in numeric_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # ===== Thay 0 bằng NaN cho Cholesterol và RestingBP =====
        for col in ['Cholesterol', 'RestingBP']:
            if col in df_processed.columns:
                zero_count = (df_processed[col] == 0).sum()
                if zero_count > 0:
                    print(f"{col}: {zero_count} giá trị 0 → sẽ thay bằng NaN")
                    df_processed[col] = df_processed[col].replace(0, np.nan)

        # ===== Điền missing bằng KNNImputer =====
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=3)
        df_processed[numeric_cols] = pd.DataFrame(
            imputer.fit_transform(df_processed[numeric_cols]),
            columns=numeric_cols,
            index=df_processed.index
        )

        # ===== Thông tin kiểm tra =====
        print("Các giá trị missing sau imputer:")
        print(df_processed[numeric_cols].isnull().sum())

        self.df_processed = df_processed


    def encode_features(self):
        df_processed = self.df_processed
        target_col1 = ['Sex', 'ExerciseAngina']
        target_col2 = ['ChestPainType', 'RestingECG', 'ST_Slope']

        # ---- LabelEncoder cho target_col1 ----
        for col in target_col1:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le

        # ---- OneHotEncoder cho target_col2 ----
        self.onehot_encoder = OneHotEncoder(drop=None, handle_unknown='ignore')
        onehot = self.onehot_encoder.fit_transform(df_processed[target_col2])
        onehot_df = pd.DataFrame(
            onehot.toarray(),
            columns=self.onehot_encoder.get_feature_names_out(target_col2),
            index=df_processed.index
        )

        df_processed = pd.concat([df_processed.drop(columns=target_col2), onehot_df], axis=1)
        self.df_processed = df_processed
        
    #Chuẩn bị dữ liệu cho model
    def prepare_for_model(self):
        df_processed = self.df_processed
        if 'HeartDisease' in df_processed.columns:
            X = df_processed.drop('HeartDisease', axis=1)
            y = df_processed['HeartDisease']
            self.X = X
            self.y = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test

            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            return None

    #tổng hợp tiền xử lý
    def run(self):
        self.clean_data()
        self.encode_features()
        return self.prepare_for_model()

    #Chuẩn hóa dữ liệu mới trc khi đưa vào model
    def transform_new_data(self, df_new: pd.DataFrame):
        if not self.label_encoders or not self.onehot_encoder or not self.scaler:
            raise ValueError("Preprocessor chưa được huấn luyện. Chạy encode_features() và prepare_for_model() trước.")

        df = df_new.copy()

        # Map số sang nhãn string cho các cột phân loại
        chest_map = {0:"TA", 1:"ATA", 2:"NAP", 3:"ASY"}
        rest_map = {0:"Normal", 1:"ST", 2:"LVH"}
        slope_map = {0:"Up", 1:"Flat", 2:"Down"}

        if "ChestPainType" in df.columns:
            df["ChestPainType"] = df["ChestPainType"].map(chest_map)
        if "RestingECG" in df.columns:
            df["RestingECG"] = df["RestingECG"].map(rest_map)
        if "ST_Slope" in df.columns:
            df["ST_Slope"] = df["ST_Slope"].map(slope_map)

        # ---- LabelEncoder cho cột nhị phân ----
        for col in ['Sex', 'ExerciseAngina']:
            le = self.label_encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # ---- OneHotEncoder ----
        onehot = self.onehot_encoder.transform(df[['ChestPainType','RestingECG','ST_Slope']])
        onehot_df = pd.DataFrame(onehot.toarray(),
                                columns=self.onehot_encoder.get_feature_names_out(['ChestPainType','RestingECG','ST_Slope']),
                                index=df.index)
        df = pd.concat([df.drop(columns=['ChestPainType','RestingECG','ST_Slope']), onehot_df], axis=1)

        # ---- Ép kiểu numeric cho các cột số ----
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ---- Chuẩn hóa ----
        return self.scaler.transform(df)


