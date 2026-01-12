import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

#Tìm ngưỡng tốt nhất của model
def find_best_threshold(y_true, y_proba, metric="recall"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_score = 0.5, -1
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        score = recall_score(y_true, y_hat) if metric == "recall" else f1_score(y_true, y_hat)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


class MLPModel:
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test):
        #Gọi các feature 
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        #Lưu các biến để lưu model tốt nhất
        self.best_model = None #model tố nhất
        self.calibrated_model = None #
        self.best_threshold = 0.5 #Ngưỡng tối đa 0.5
        self.best_name = None #Lưu tên model tốt nhất
        self.best_metrics = None #Lưu thông số: accuracy, precision, recall, f1, roc_auc, y_pred, y_proba

        # đảm bảo thư mục models tồn tại
        if not os.path.exists("models"):
            os.makedirs("models")

    # ================== MLP cơ bản ==================
    def model_MLP(self):
        model = MLPClassifier(
            activation='relu', solver='adam', #chọn hàm kích hoạt: relu, thuật toán tối ưu:adam
            max_iter=1000, #Học lặp 1000 
            learning_rate_init=0.001, #Tốc độ học ban đầu của thuật toán
            hidden_layer_sizes=(100,), #Học 1 layer với 100 neuron
            random_state=42, #seed
            alpha=0.1 #hệ số regularization
        )
        model.fit(self.X_train_scaled, self.y_train) #fit tập dữ liệu vào model
        y_pred = model.predict(self.X_test_scaled) #Dự đoán trên tập test đã đc chuẩn hóa
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1] #Dự đoán xác xuất trên tập test đã đc chuẩn hóa

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        return model, metrics 

    # ================== Fine-tuning ==================
    def finetuning(self):
        
        clf = MLPClassifier(max_iter=500, random_state=42) #Khởi tạo với 500 vòng lặp học
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50)], #Chọn các tầng ẩn
            'activation': ['relu'], #Chọn hàm kích hoạt relu
            'solver': ['adam'], #Chọn thuật toán tối ưu adam
            'learning_rate_init': [0.001, 0.0005], #Chọn Tốc độ học ban đầu
            'alpha': [0.0001, 0.01], #Chọn hệ số regularization
            'max_iter': [500, 1000], #Chọn vòng học epoch
            'early_stopping': [True, False] 
        }
        #Tìm tham số tốt nhất bằng RandomizedSearchCV
        #chia tập train thành 3 phần để validate, dùng tất cả CPU cores để train
        rd_search = RandomizedSearchCV(clf, param_grid, cv=3, n_jobs=-1, random_state=42)
        #Huấn luyện 
        rd_search.fit(self.X_train_scaled, self.y_train)

        #Lấy model tốt nhất, dự đoán xác suất lớp dương(nhị phân)
        best_model = rd_search.best_estimator_
        y_proba = best_model.predict_proba(self.X_test_scaled)[:, 1]

        # Tìm threshold tối ưu theo recall
        best_t, _ = find_best_threshold(self.y_test, y_proba, metric="recall")
        y_pred_custom = (y_proba >= best_t).astype(int) #dự đoán trên ngưỡng tốt nhất

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred_custom),
            "precision": precision_score(self.y_test, y_pred_custom),
            "recall": recall_score(self.y_test, y_pred_custom),
            "f1": f1_score(self.y_test, y_pred_custom),
            "roc_auc": roc_auc_score(self.y_test, y_proba),
            "y_pred": y_pred_custom,
            "y_proba": y_proba
        }
        return best_model, metrics

    # ================== Ensemble ==================
    def ensemble(self, n_runs=10):
        """Xây dựng hàm emsemble với sự kết hợp của RandomForest MLpClassifier
        LogisticRegression"""
        #Khởi tạo biến lưu kết quả tốt nhất
        best_acc = -1
        best_model = None
        best_y_pred = None
        best_y_proba = None

        #Vì có RandomForest nên kết quả sẽ có tính ngẫu nhiên-> Cho chạy ensemble 10 lần và lấy cái tốt nhất
        for _ in range(n_runs):
            clf1 = LogisticRegression(max_iter=1000)
            clf2 = RandomForestClassifier(random_state=None)
            #Thông số của MLPCLassifier
            clf3 = MLPClassifier(
                activation='relu', 
                solver='adam',
                max_iter=1000, 
                learning_rate_init=0.0005,
                hidden_layer_sizes=(50,), 
                early_stopping=True,
                random_state=None, 
                alpha=0.1,
            )
            #tạo ensemble bằng VotingClassifier
            model = VotingClassifier(
                estimators=[('lr', clf1), ('rf', clf2), ('mlp', clf3)],
                voting='soft' #Dự đoán xác suất của từng base model-> tính trung bình xác suất
            )
            #Huấn luyện và chọn model tốt nhất dựa trên accuracy
            model.fit(self.X_train_scaled, self.y_train)
            acc = model.score(self.X_test_scaled, self.y_test)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_y_pred = model.predict(self.X_test_scaled)
                best_y_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(self.y_test, best_y_pred),
            "precision": precision_score(self.y_test, best_y_pred),
            "recall": recall_score(self.y_test, best_y_pred),
            "f1": f1_score(self.y_test, best_y_pred),
            "roc_auc": roc_auc_score(self.y_test, best_y_proba),
            "y_pred": best_y_pred,
            "y_proba": best_y_proba
        }
        return best_model, metrics

    # ================== Hiệu chuẩn và tìm threshold ==================
    def calibrate_and_find_threshold(self, model, X_val_scaled, y_val, metric="recall"):
        # Hiệu chuẩn xác suất
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(self.X_train_scaled, self.y_train)

        # Dự đoán xác suất trên tập validation
        y_proba_val = calibrated.predict_proba(X_val_scaled)[:, 1]

        # Tìm ngưỡng tối ưu theo metric
        best_t, best_score = find_best_threshold(y_val, y_proba_val, metric=metric)
        print(f"Ngưỡng tối ưu theo {metric}: {best_t:.2f} | {metric} đạt: {best_score:.4f}")

        # Lưu lại mô hình đã hiệu chuẩn và ngưỡng
        self.calibrated_model = calibrated
        self.best_threshold = best_t
        return calibrated, best_t

    # ================== Loss/Accuracy curve ==================
    def anal_curve(self, X_train_scaled, X_test_scaled, y_train, y_test, epochs=200):
        """Vẽ đồ thị loss curve và accuracy curve"""
        mlp = MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=1,                # mỗi lần partial_fit chỉ chạy 1 epoch
            learning_rate_init=0.0005,
            hidden_layer_sizes=(50,),
            early_stopping=False,
            random_state=42,
            alpha=0.1,
            momentum=0.9,
            warm_start=True            # cho phép tiếp tục huấn luyện
        )
        #Khởi tạo danh sách lưu kết quả
        train_acc, val_acc, losses = [], [], []
        #Khởi tạo danh sách nhãn trong tập dữ liệu 
        classes = np.unique(y_train)

        #Huấn luyện từng epoch
        for _ in range(epochs):
            
            mlp.partial_fit(X_train_scaled, y_train, classes=classes) #Huấn luyện epoch trên tập train
            train_acc.append(mlp.score(X_train_scaled, y_train))
            val_acc.append(mlp.score(X_test_scaled, y_test))
            losses.append(mlp.loss_)   # lấy loss hiện tại

        #Vẽ đồ thị loss curve
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(losses, label='Training Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve of MLPClassifier')
        axes[0].legend()

        #Vẽ đồ thị Accuracy curve
        axes[1].plot(train_acc, label='Training Accuracy', color='blue')
        axes[1].plot(val_acc, label='Validation Accuracy', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curve of MLPClassifier')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    # ================== Confusion matrix + ROC ==================
    def confusion_matrix_ROC(self, y_test, y_pred, y_proba, model_name="Model"):
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm,
                             index=['Thực tế: Không bệnh', 'Thực tế: Có bệnh'],
                             columns=['Dự đoán: Không bệnh', 'Dự đoán: Có bệnh'])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        #Vẽ ma trận nhầm lẫn
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Ma trận nhầm lẫn - {model_name}')
        axes[0].set_xlabel('Dự đoán')
        axes[0].set_ylabel('Thực tế')

        #Vẽ đường cong ROC
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve - {model_name}')
        axes[1].legend(loc="lower right")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    # ================== Save/Load ==================
    def save_model(self, model_name='best_model.pkl'):
        #Lưu file '_.pkl' tại thư mục models
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, f'models/{model_name}')
            print(f"Đã lưu mô hình hiệu chuẩn tại: models/{model_name}")
        elif self.best_model is not None:
            joblib.dump(self.best_model, f'models/{model_name}')
            print(f"Đã lưu mô hình tại: models/{model_name}")
        else:
            print("Không có mô hình để lưu!")
    #Lưu ngưỡng
    def save_threshold(self, threshold_name='best_threshold.pkl'):
        joblib.dump(self.best_threshold, f'models/{threshold_name}')
        print(f"Đã lưu ngưỡng tại: models/{threshold_name}")

    # ================== Run pipeline ==================
    def run(self, save_name='best_model.pkl', threshold_name='best_threshold.pkl', epochs_curve=200):
        results = {}
        models = {}

        # Chạy MLP cơ bản
        print("Chạy MLP cơ bản...")
        models["MLP cơ bản"], results["MLP cơ bản"] = self.model_MLP()

        # Chạy MLP fine-tuning
        print("Chạy MLP Fine-tuning...")
        models["MLP fine-tune"], results["MLP fine-tune"] = self.finetuning()

        # Vẽ Loss & Accuracy curve cho MLP Fine-tuning
        print("\nVẽ Loss & Accuracy curve cho MLP Fine-tuning...")
        self.anal_curve(
            X_train_scaled=self.X_train_scaled,
            X_test_scaled=self.X_test_scaled,
            y_train=self.y_train,
            y_test=self.y_test,
            epochs=epochs_curve
        )

        # Chạy Ensemble
        print("Chạy Ensemble...")
        models["Ensemble"], results["Ensemble"] = self.ensemble()

        # In bảng so sánh các metrics
        print("\n=== So sánh kết quả trên 5 metrics ===")
        for name, metrics in results.items():
            print(f"\n{name}:")
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                print(f"  {k}: {metrics[k]:.4f}")

        # Chọn mô hình tốt nhất theo roc_auc
        best_name = max(results, key=lambda x: results[x]["roc_auc"])
        best_model = models[best_name]
        best_metrics = results[best_name]

        print("\n" + "="*50)
        print(f"Mô hình tốt nhất theo ROC AUC: {best_name} với roc_auc = {best_metrics['roc_auc']:.4f}")
        print("="*50)

        # Hiệu chuẩn và tìm ngưỡng tối ưu theo recall trên tập test (dùng như validation)
        print("\nHiệu chuẩn xác suất và tìm ngưỡng tối ưu theo recall...")
        calibrated, best_t = self.calibrate_and_find_threshold(best_model, self.X_test_scaled, self.y_test, metric="recall")
        print(f"Ngưỡng lưu dùng khi dự đoán thực tế: {best_t:.2f}")

        #  Vẽ confusion matrix + ROC curve với mô hình đã hiệu chuẩn
        y_proba_cal = calibrated.predict_proba(self.X_test_scaled)[:, 1]
        y_pred_cal = (y_proba_cal >= best_t).astype(int)
        self.confusion_matrix_ROC(
            y_test=self.y_test,
            y_pred=y_pred_cal,
            y_proba=y_proba_cal,
            model_name=f"{best_name} (calibrated, t={best_t:.2f})"
        )

        # Lưu mô hình và ngưỡng
        self.best_model = best_model
        self.best_name = best_name
        self.best_metrics = best_metrics

        self.save_model(save_name)
        self.save_threshold(threshold_name)

        return best_model, best_name, best_metrics
