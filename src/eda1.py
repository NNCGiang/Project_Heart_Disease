import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def eda_overview(self):
        print("\n" + "="*50)
        print("THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU")
        print("="*50)

        print("\n 5 DÒNG ĐẦU TIÊN:")
        print(self.df.head())

        print("\n 5 DÒNG CUỐI CÙNG:")
        print(self.df.tail())

        print("\nTHÔNG TIN CƠ BẢN:")
        self.df.info()

        print("\n THỐNG KÊ MÔ TẢ (SỐ):")
        print(self.df.describe().T)

        print("\n THỐNG KÊ MÔ TẢ (PHÂN LOẠI):")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
            print(f"Số lượng duy nhất: {self.df[col].nunique()}")
            
    #Phân tích dữ liệu thiếu
    def missingvalue(self):
        print("\n" + "="*50)
        print(" PHÂN TÍCH GIÁ TRỊ THIẾU (MISSING VALUES)")
        print("="*50)

        #Đếm số lượng giá trị thiếu và phần trăm
        missing_values = self.df.isnull().sum()
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        #In ra số lượng và phần trăm giá trị bị thiếu
        missing_df = pd.DataFrame({
            'Số lượng thiếu': missing_values,
            'Phần trăm thiếu': missing_percent
        })

        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0]
        
        #Nếu có thì in ra giá trị thiếu
        if len(missing_df) > 0:
            print(" CÓ GIÁ TRỊ THIẾU:")
            print(missing_df)
        else:
            print(" KHÔNG CÓ GIÁ TRỊ THIẾU!")

    #Phân tích giá trị ngoại lai
    def outlier(self):
        print("\n" + "="*50)
        print(" PHÂN TÍCH NGOẠI LAI (OUTLIERS)")
        print("="*50)

        #Lấy các features số
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(2, 4, figsize=(10, 6))
        axes = axes.ravel()

        #Vẽ biểu đồ ngoại lai của các features
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                self.df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Boxplot của {col}')
                axes[i].set_ylabel('Giá trị')

        plt.tight_layout()
        plt.show()

        #Phân tích ngoại lai bằng IQR
        print("\n PHÂN TÍCH OUTLIERS BẰNG IQR:")
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")

    #Phân tích dữ liệu
    def target_anal(self):
        candidate_targets = ["DEATH_EVENT", "HeartDisease", "target", "Outcome"]
        target = next((c for c in candidate_targets if c in self.df.columns), None)
        assert target is not None, f"No known target found. Columns: {list(self.df.columns)}"

        print(f"Using target: {target}\n")
        print("Dtypes:\n", self.df.dtypes, "\n")
        print("Missing values per column:\n", self.df.isna().sum(), "\n")
        print("Target counts:\n", self.df[target].value_counts())
        print("Target proportion:\n", self.df[target].value_counts(normalize=True).round(3))

        plot_cols = [c for c in ["Age","age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease", "ejection_fraction"] if c in self.df.columns]

        fig, axes = plt.subplots(1, 1 + len(plot_cols), figsize=(4 * (1 + len(plot_cols)), 3))
        ax0 = axes if len(plot_cols) == 0 else axes[0]

        sns.countplot(x=target, data=self.df, ax=ax0)
        ax0.set_title(target)

        if len(plot_cols) > 0:
            for i, col in enumerate(plot_cols, start=1):
                sns.histplot(self.df[col], bins=20, ax=axes[i])
                axes[i].set_title(col)

        plt.tight_layout()
        plt.show()

    #Phân tích features heartdisease
    def heartdisease_anal(self):
        print("\n" + "="*50)
        print(" PHÂN TÍCH BIẾN MỤC TIÊU (HeartDisease)")
        print("="*50)

        #Nếu có cột HeartDisease thì vẽ biểu đồ
        if 'HeartDisease' in self.df.columns:
            plt.ioff()
            plt.figure(figsize=(10, 6))
            heart_disease_counts = self.df['HeartDisease'].value_counts()
            colors = ['#ff9999', '#66b3ff']
            plt.pie(heart_disease_counts.values,
                    labels=[f'Có bệnh (1)', f'Không bệnh (0)'],
                    autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Phân phối Biến Mục tiêu - HeartDisease')
            plt.show()

            print(f"\n Số lượng cụ thể:")
            print(f"• Có bệnh tim (1): {heart_disease_counts.get(1, 0)} ({heart_disease_counts.get(1, 0)/len(self.df)*100:.1f}%)")
            print(f"• Không bệnh tim (0): {heart_disease_counts.get(0, 0)} ({heart_disease_counts.get(0, 0)/len(self.df)*100:.1f}%)")
        else:
            print(" Không tìm thấy cột 'HeartDisease' trong dữ liệu")

    #Phân tích Tỷ lệ mắc vấn đề về tim theo giới tính
    def sex_anal(self):
        print("\n" + "="*50)
        print(" PHÂN TÍCH THEO GIỚI TÍNH")
        print("="*50)

        #Có features 'sex' thì vẽ biểu đồ
        if 'Sex' in self.df.columns:
            gender_counts = self.df['Sex'].value_counts()
            print(f"\n Phân phối giới tính:")
            print(f"• Nam (M): {gender_counts.get('M', 0)} ({(gender_counts.get('M', 0)/len(self.df))*100:.1f}%)")
            print(f"• Nữ (F): {gender_counts.get('F', 0)} ({(gender_counts.get('F', 0)/len(self.df))*100:.1f}%)")

            if 'HeartDisease' in self.df.columns:
                gender_disease = pd.crosstab(self.df['Sex'], self.df['HeartDisease'], normalize='index') * 100
                plt.figure(figsize=(10, 6))
                gender_disease.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
                plt.title('TỶ LỆ BỆNH TIM THEO GIỚI TÍNH')
                plt.xlabel('Giới tính')
                plt.ylabel('Phần trăm (%)')
                plt.legend(['Không bệnh', 'Có bệnh'])
                plt.xticks(rotation=0)
                plt.show()

                print("\n Tỷ lệ bệnh theo giới tính:")
                print(gender_disease)
        else:
            print(" Không tìm thấy cột 'Sex' trong dữ liệu")

    #Phân tích tỷ lệ mắc vấn đề về tim theo độ tuổi
    def age_anal(self):
        print("\n" + "="*50)
        print(" PHÂN TÍCH THEO ĐỘ TUỔI")
        print("="*50)

        if 'Age' in self.df.columns:
            # Vẽ histplot + boxplot (chỉ 1 lần)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(data=self.df, x='Age', hue='HeartDisease', bins=30, kde=True, palette='Set2', ax=axes[0])
            axes[0].set_title('Phân phối Tuổi theo Tình trạng Bệnh')

            sns.boxplot(data=self.df, x='HeartDisease', y='Age', palette='Set2', ax=axes[1])
            axes[1].set_title('Boxplot Tuổi theo Tình trạng Bệnh')

            plt.tight_layout()
            plt.show()

            # Phân nhóm tuổi
            self.df['AgeGroup'] = pd.cut(
                self.df['Age'],
                bins=[0, 30, 40, 50, 60, 70, 100],
                labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+']
            )

            if 'HeartDisease' in self.df.columns:
                agegroup_disease = pd.crosstab(self.df['AgeGroup'], self.df['HeartDisease'], normalize='index') * 100
                print("\n Tỷ lệ bệnh theo nhóm tuổi:")
                print(agegroup_disease)
        else:
            print(" Không tìm thấy cột 'Age' trong dữ liệu")
    #Vẽ ma trận tương quan giữa các features
    def heatmap(self, df, target='HeartDisease'):
        print("\n" + "="*50)
        print(" PHÂN TÍCH TƯƠNG QUAN VÀ MỐI QUAN HỆ")
        print("="*50)

        # Ma trận tương quan
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', linewidths=0.5)
            plt.title('MA TRẬN TƯƠNG QUAN GIỮA CÁC BIẾN SỐ')
            plt.show()

            # Tương quan với biến mục tiêu
            if 'HeartDisease' in correlation_matrix.columns:
                print("\n TƯƠNG QUAN VỚI BIẾN MỤC TIÊU (HeartDisease):")
                correlation_with_target = correlation_matrix['HeartDisease'].sort_values(ascending=False)
                for idx, val in correlation_with_target.items():
                    if idx != 'HeartDisease':
                        print(f"{idx}: {val:.3f}")
                

  
