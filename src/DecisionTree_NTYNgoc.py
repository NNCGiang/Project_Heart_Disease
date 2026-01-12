import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV
import joblib

class DecisionTreePipeline:
    ## Phần chuẩn bị & khởi tạo pipeline
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names=None):
        self.X_train_scaled = X_train_scaled #Lưu tập dữ liệu huấn luyện (đã được chuẩn hoá)
        self.X_test_scaled = X_test_scaled #Lưu tập dữ liệu kiểm tra
        self.y_train = y_train #Lưu nhãn của tập huấn luyện (0, 1)
        self.y_test = y_test #Lưu nhãn dùng để đánh giá
        self.feature_names = feature_names #Lưu tên các cột đặc trưng
        self.model = None #Lưu mô hình cơ bản
        self.best_model = None #Lưu mô hình tốt nhất sau tuning


    ## Phần xây dựng và đánh giá mô hình
    def train_decision_tree(self
                            , max_depth=3 #Giới hạn độ sâu của cây tránh overfitting
                            , random_state=42 #Cố định kết quả sinh ngẫu nhiên
                            ):
        """Huấn luyện mô hình Decision Tree cơ bản"""
        self.model = DecisionTreeClassifier(max_depth=max_depth #Kiểm soát độ phức tạp của cây
                                            , random_state=random_state #Đảm bảo kết quả ổn định
                                            )
        self.model.fit(self.X_train_scaled, self.y_train)
        return self.model
    ## Phần đánh giá mô hình
    def evaluate_model(self, model=None):
        """Đánh giá mô hình với các metrics"""
        if model is None:
            model = self.model
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred), #Tỷ lệ dự đoán đúng trên tổng số mẫu
            "precision": precision_score(self.y_test, y_pred), #trong số dự đoán có bệnh, bao nhiêu ca đúng
            "recall": recall_score(self.y_test, y_pred), #mô hình phát hiện bao nhiêu người trong số người 1
            "f1": f1_score(self.y_test, y_pred), #đánh giá cân bằng mô hình (precision và recall)
            "roc_auc": roc_auc_score(self.y_test, y_proba) if y_proba is not None else None, # khả năng phân biệt 0 và 1
            "y_pred": y_pred, 
            "y_proba": y_proba
        }
        return metrics


    ## Phần Confusion Matrix
    def plot_confusion_matrix(self, y_pred, title="Ma trận nhầm lẫn"):
        cm = confusion_matrix(self.y_test, y_pred) # tạo ma trận 
        plt.figure(figsize=(6, 5)) #Tạo khung hình cho biểu đồ
        sns.heatmap(cm, annot=True #Hiển thị giá trị trong từng ô
                    , fmt='d' #Hiển thị số nguyên
                    , cmap='Blues'
                    )
        plt.title(title)
        plt.xlabel('Giá trị dự đoán')
        plt.ylabel('Giá trị thực tế')
        plt.show()


    ## Phần vẽ cây
    def plot_decision_tree(self, model=None):
        #Nếu không truyền mô hình vào hàm thì dùng mô hình đã train trước đó
        if model is None:
            model = self.model
        plt.figure(figsize=(22, 12))
        plot_tree(model,
                  feature_names=self.feature_names if self.feature_names is not None else None, #giúp hiện thị ra tên thuộc tính thay vì X[]
                  class_names=['Không bệnh', 'Có bệnh'],
                  filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree Classifier")
        plt.show()


    ## Phần tuning
    def tune_max_depth(self, max_depth_values=range(3, 20)):
        best_score = 0.0 #Lưu độ chính xác cao nhất
        best_depth = None #Lưu giá trị max_depth tương ứng
        for depth in max_depth_values:
            # Khởi tạo mô hình Decision Tree với độ sâu hiện tại
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

            # Huấn luyện mô hình trên tập train
            clf.fit(self.X_train_scaled, self.y_train)

            # Đánh giá mô hình trên tập test
            score = clf.score(self.X_test_scaled, self.y_test)
            print(f"Max Depth = {depth}: Test Accuracy = {score:.4f}")

            #Nếu Accuracy cao hơn giá trị tốt nhất trước đó thì cập nhật
            if score > best_score:
                best_score = score
                best_depth = depth
        print(f"\nBest Max_depth = {best_depth} : Best Accuracy = {best_score:.4f}")
        return best_depth, best_score


    ## Phần tunning nâng cao
    def grid_search_decision_tree(self):
        dt = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 8, 10, None], #Độ sâu tối đa của cây
            'min_samples_split': [2, 5, 10, 20], # Số mẫu tối thiểu để tách 1 node
            'min_samples_leaf': [1, 2, 5, 10], #Số mẫu tối thiểu tại node lá
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train_scaled, self.y_train)
        print("Thông số tốt nhất:", grid_search.best_params_)
        print("Độ chính xác CV tốt nhất:", grid_search.best_score_)
        self.best_model = grid_search.best_estimator_ #Lưu lại mô hình tốt nhất
        return self.best_model


    ## Phần biểu đồ Accuracy vs Max Deth
    def plot_accuracy_vs_depth(self, depths=range(1, 21)):
        train_acc, val_acc = [], []
        for depth in depths:
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(self.X_train_scaled, self.y_train)

            #Accuracy trên tập huấn luyện
            train_acc.append(dt.score(self.X_train_scaled, self.y_train))

            #Accuracy trên tập kiểm tra
            val_acc.append(dt.score(self.X_test_scaled, self.y_test))
        plt.figure(figsize=(12, 5))
        plt.plot(depths, train_acc, label='Độ chính xác tập huấn luyện')
        plt.plot(depths, val_acc, label='Độ chính xác tập kiểm tra')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree – Accuracy vs Max Depth')
        plt.legend()
        plt.grid(True)
        plt.show()


    ## Phần ROC_Curve (Đánh giá khả năng phân biệt)
    def plot_roc_curve(self, y_proba, model_name="Decision Tree"):

        ## Tính False Positive Rate (FPR) và True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)

        # Tính diện tích dưới đường cong ROC (AUC)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})') #Vẽ đường cong ROC
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') #vẽ đường chéo
        plt.xlabel('Tỷ lệ dương tính giả (FPR)')
        plt.ylabel('Tỷ lệ dương tính thật (TPR)')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    # ================== Save/Load ==================
    def save_model(self, model_name='best_model.pkl'):
        #Lưu file '_.pkl' tại thư mục models
        if self.best_model is not None:
            joblib.dump(self.best_model, f'models/{model_name}')
            print(f"Đã lưu mô hình tại: models/{model_name}")
        else:
            print("Không có mô hình để lưu!")
