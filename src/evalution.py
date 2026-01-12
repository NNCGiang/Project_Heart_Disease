import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, y_test, results_dict):
        """
        results_dict: dict chứa kết quả của các mô hình
        """
        self.y_test = y_test
        self.results_dict = results_dict
        # Lọc các cột chỉ numeric
        metrics_only = {}
        for m, res in results_dict.items():
            metrics_only[m] = {}
            for k, v in res.items():
                if k not in ["y_pred","y_proba"]:
                    try:
                        metrics_only[m][k] = float(v)
                    except:
                        pass  # loại bỏ nếu không convert được

        self.df_results = pd.DataFrame(metrics_only).T
    
    #Vẽ biểu đồ so sánh metrics
    def plot_bar_metrics(self):
        """Vẽ biểu đồ cột so sánh các metrics"""
        plt.figure(figsize=(10,6))
        self.df_results.plot(kind='bar')
        plt.title("So sánh metrics giữa các mô hình")
        plt.ylabel("Giá trị")
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    #Vẽ ma trận tương quan 3 model
    def plot_heatmap(self):
        """Vẽ heatmap so sánh"""
        plt.figure(figsize=(8,6))
        sns.heatmap(self.df_results, annot=True, cmap="coolwarm", fmt=".3f")
        plt.title("Heatmap so sánh mô hình")
        plt.show()
        
    #Lưu model tốt 
    def best_model(self, metric="accuracy"):
        """Trả về mô hình tốt nhất theo metric"""
        best_name = self.df_results[metric].idxmax()
        best_score = self.df_results[metric].max()
        print(f"Mô hình tốt nhất theo {metric}: {best_name} ({best_score:.4f})")
        return best_name, best_score
    
    #Vẽ ma trận nhầm lẫn
    def plot_confusion_matrices(self):
        """Vẽ Confusion Matrix của tất cả mô hình cạnh nhau"""
        n_models = len(self.results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]  # nếu chỉ có 1 mô hình

        for ax, (model_name, metrics) in zip(axes, self.results_dict.items()):
            y_pred = metrics["y_pred"]
            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        plt.tight_layout()
        plt.show()
        
    #Vẽ đường ROC của 3 model
    def plot_roc_all_models(self):
        """Vẽ ROC Curve của tất cả mô hình trong cùng một cửa sổ"""
        plt.figure(figsize=(8,6))
        for model_name, metrics in self.results_dict.items():
            y_proba = metrics["y_proba"]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        # Đường chéo tham chiếu
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("So sánh ROC Curve giữa các mô hình")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

