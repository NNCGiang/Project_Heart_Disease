import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.inspection import permutation_importance

class GradientBoostingModel:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names=None, save_dir='models'):
        """
        Khởi tạo model Gradient Boosting
        
        Parameters:
        -----------
        X_train, X_test: Dữ liệu đặc trưng đã chuẩn hóa
        y_train, y_test: Nhãn tương ứng
        feature_names: Tên các đặc trưng (nếu không có sẽ tự động tạo)
        save_dir: Thư mục lưu model và dữ liệu
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X_train.shape[1])]
        self.save_dir = save_dir
        
        self.best_model = None
        self.best_name = None
        self.best_metrics = None
        self.feature_importance = None
        self.all_results = {}
        self.all_models = {}
        
        # Tạo thư mục lưu trữ nếu chưa có
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'datasets'), exist_ok=True)
    
    # ==================================================
    # HÀM LƯU DATASET
    # ==================================================
    def save_datasets(self, prefix='gb'):
        """
        Lưu các dataset đã chia (train/test) để sử dụng sau này
        """
        try:
            # Lưu dữ liệu train
            train_data = pd.DataFrame(self.X_train, columns=self.feature_names)
            train_data['target'] = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
            train_path = os.path.join(self.save_dir, 'datasets', f'{prefix}_train_dataset.csv')
            train_data.to_csv(train_path, index=False)
            
            # Lưu dữ liệu test
            test_data = pd.DataFrame(self.X_test, columns=self.feature_names)
            test_data['target'] = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            test_path = os.path.join(self.save_dir, 'datasets', f'{prefix}_test_dataset.csv')
            test_data.to_csv(test_path, index=False)
            
            # Lưu thông tin dataset
            dataset_info = {
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape,
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'n_features': self.X_train.shape[1],
                'train_path': train_path,
                'test_path': test_path,
                'feature_names': self.feature_names,
                'target_distribution_train': pd.Series(self.y_train).value_counts().to_dict(),
                'target_distribution_test': pd.Series(self.y_test).value_counts().to_dict()
            }
            
            info_path = os.path.join(self.save_dir, 'datasets', f'{prefix}_dataset_info.pkl')
            joblib.dump(dataset_info, info_path)
            
            print("\n" + "="*60)
            print("ĐÃ LƯU DATASETS THÀNH CÔNG!")
            print("="*60)
            print(f"Train dataset: {train_path}")
            print(f"Test dataset: {test_path}")
            print(f"Dataset info: {info_path}")
            print(f"\nThông tin dataset:")
            print(f"  Train shape: {self.X_train.shape}")
            print(f"  Test shape: {self.X_test.shape}")
            print(f"  Số đặc trưng: {self.X_train.shape[1]}")
            
            return train_path, test_path, info_path
            
        except Exception as e:
            print(f"Lỗi khi lưu datasets: {e}")
            return None, None, None
    
    def load_datasets(self, train_path, test_path):
        """
        Tải datasets đã lưu
        """
        try:
            # Tải train dataset
            train_data = pd.read_csv(train_path)
            self.X_train = train_data.drop('target', axis=1).values
            self.y_train = train_data['target'].values
            self.feature_names = train_data.drop('target', axis=1).columns.tolist()
            
            # Tải test dataset
            test_data = pd.read_csv(test_path)
            self.X_test = test_data.drop('target', axis=1).values
            self.y_test = test_data['target'].values
            
            print("\n" + "="*60)
            print("ĐÃ TẢI DATASETS THÀNH CÔNG!")
            print("="*60)
            print(f"Train dataset: {train_path}")
            print(f"Test dataset: {test_path}")
            print(f"Train shape: {self.X_train.shape}")
            print(f"Test shape: {self.X_test.shape}")
            print(f"Số đặc trưng: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi tải datasets: {e}")
            return False
    
    # ==================================================
    # HÀM HUẤN LUYỆN MODEL CƠ BẢN
    # ==================================================
    def model_gb_basic(self):
        """
        Gradient Boosting với tham số cơ bản
        """
        print("\n[1/6] Training Gradient Boosting cơ bản...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        
        # Lưu feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   ✓ GB cơ bản - Accuracy: {metrics['accuracy']:.4f}")
        return model, metrics
    
    def model_gb_advanced(self):
        """
        Gradient Boosting với tham số nâng cao
        """
        print("\n[2/6] Training Gradient Boosting nâng cao...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        print(f"   ✓ GB nâng cao - Accuracy: {metrics['accuracy']:.4f}")
        return model, metrics
    
    # ==================================================
    # HÀM TINH CHỈNH THAM SỐ
    # ==================================================
    def finetuning_randomized(self):
        """
        Tinh chỉnh tham số với RandomizedSearchCV
        """
        print("\n[3/6] Tinh chỉnh tham số với RandomizedSearchCV...")
        
        param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            gb, param_dist, n_iter=20, cv=3, 
            scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        metrics["best_params"] = random_search.best_params_
        
        print(f"   ✓ Best params: {random_search.best_params_}")
        print(f"   ✓ GB RandomizedSearch - Accuracy: {metrics['accuracy']:.4f}")
        return best_model, metrics
    
    def finetuning_grid(self):
        """
        Tinh chỉnh tham số với GridSearchCV
        """
        print("\n[4/6] Tinh chỉnh tham số với GridSearchCV...")
        
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        metrics["best_params"] = grid_search.best_params_
        
        print(f"   ✓ Best params: {grid_search.best_params_}")
        print(f"   ✓ GB GridSearch - Accuracy: {metrics['accuracy']:.4f}")
        return best_model, metrics
    
    # ==================================================
    # HÀM ENSEMBLE
    # ==================================================
    def ensemble_gb(self, n_runs=5):
        """
        Ensemble với Gradient Boosting và các model khác
        """
        print(f"\n[5/6] Training Ensemble model với {n_runs} runs...")
        
        best_acc = 0
        best_model = None
        best_y_pred = None
        best_y_proba = None
        
        for i in range(n_runs):
            # Các base models
            gb1 = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=42+i
            )
            
            gb2 = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=5,
                random_state=42+i*2
            )
            
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=5,
                random_state=42+i*3
            )
            
            lr = LogisticRegression(
                max_iter=1000, random_state=42+i*4
            )
            
            # Ensemble với voting
            ensemble_model = VotingClassifier(
                estimators=[
                    ('gb1', gb1),
                    ('gb2', gb2),
                    ('rf', rf),
                    ('lr', lr)
                ],
                voting='soft'
            )
            
            # Train và evaluate
            ensemble_model.fit(self.X_train, self.y_train)
            acc = ensemble_model.score(self.X_test, self.y_test)
            
            if acc > best_acc:
                best_acc = acc
                best_model = ensemble_model
                best_y_pred = ensemble_model.predict(self.X_test)
                best_y_proba = ensemble_model.predict_proba(self.X_test)[:, 1]
                
                print(f"   Run {i+1}: New best accuracy = {acc:.4f}")
        
        print(f"\n   ✓ Ensemble tốt nhất sau {n_runs} runs có accuracy = {best_acc:.4f}")
        
        metrics = self._calculate_metrics(self.y_test, best_y_pred, best_y_proba)
        return best_model, metrics
    
    def stacking_gb(self):
        """
        Stacking với Gradient Boosting làm meta-learner
        """
        print("\n[6/6] Training Stacking model...")
        
        from sklearn.ensemble import StackingClassifier
        
        # Base models
        base_models = [
            ('gb1', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, 
                max_depth=3, random_state=42
            )),
            ('gb2', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=5, random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=5,
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=1000, random_state=42
            ))
        ]
        
        # Meta-learner
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3,
            passthrough=False
        )
        
        stacking_model.fit(self.X_train, self.y_train)
        y_pred = stacking_model.predict(self.X_test)
        y_proba = stacking_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        print(f"   ✓ Stacking - Accuracy: {metrics['accuracy']:.4f}")
        return stacking_model, metrics
    
    # ==================================================
    # HÀM HỖ TRỢ
    # ==================================================
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """
        Tính toán các metrics đánh giá
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
    
    # ==================================================
    # HÀM CHÍNH CHẠY TẤT CẢ MODEL
    # ==================================================
    def run_models(self, use_ensemble=True, use_stacking=True, save_datasets=True):
        """
        Chạy tất cả các model và so sánh kết quả
        
        Parameters:
        -----------
        use_ensemble: Có sử dụng ensemble model không
        use_stacking: Có sử dụng stacking model không
        save_datasets: Có lưu datasets không
        """
        print("\n" + "="*80)
        print("BẮT ĐẦU HUẤN LUYỆN CÁC MÔ HÌNH GRADIENT BOOSTING")
        print("="*80)
        
        # Lưu datasets nếu được yêu cầu
        if save_datasets:
            self.save_datasets()
        
        # Chạy các model
        self.all_models["GB Basic"], self.all_results["GB Basic"] = self.model_gb_basic()
        self.all_models["GB Advanced"], self.all_results["GB Advanced"] = self.model_gb_advanced()
        self.all_models["GB Randomized"], self.all_results["GB Randomized"] = self.finetuning_randomized()
        self.all_models["GB GridSearch"], self.all_results["GB GridSearch"] = self.finetuning_grid()
        
        if use_ensemble:
            self.all_models["GB Ensemble"], self.all_results["GB Ensemble"] = self.ensemble_gb()
        
        if use_stacking:
            self.all_models["GB Stacking"], self.all_results["GB Stacking"] = self.stacking_gb()

        # Hiển thị bảng so sánh
        self._display_comparison_table()
        
        # Chọn model tốt nhất
        self.best_name = max(self.all_results, key=lambda x: self.all_results[x]["accuracy"])
        self.best_model = self.all_models[self.best_name]
        self.best_metrics = self.all_results[self.best_name]
        
        # Hiển thị kết quả tốt nhất
        self._display_best_model()
        
        # Visualize
        self.confusion_matrix_ROC()
        self.plot_feature_importance()
        self.plot_learning_curve()
        
        # Lưu kết quả so sánh
        self.save_comparison_results()
        
        return self.best_model, self.best_name, self.best_metrics
    
    def _display_comparison_table(self):
        """
        Hiển thị bảng so sánh kết quả
        """
        print("\n" + "="*80)
        print("BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
        print("="*80)
        
        # Tạo DataFrame
        comparison_data = []
        
        for name, metrics in self.all_results.items():
            row = {
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            }
            
            if 'best_params' in metrics:
                # Rút gọn tham số để hiển thị
                params_str = str(metrics['best_params'])
                if len(params_str) > 50:
                    params_str = params_str[:47] + "..."
                row['Best Params'] = params_str
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("="*80)
    
    def _display_best_model(self):
        """
        Hiển thị thông tin model tốt nhất
        """
        print("\n" + "="*80)
        print("MÔ HÌNH TỐT NHẤT")
        print("="*80)
        print(f"Tên: {self.best_name}")
        print(f"Loại: {type(self.best_model).__name__}")
        print(f"Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"Precision: {self.best_metrics['precision']:.4f}")
        print(f"Recall: {self.best_metrics['recall']:.4f}")
        print(f"F1-Score: {self.best_metrics['f1']:.4f}")
        print(f"ROC-AUC: {self.best_metrics['roc_auc']:.4f}")
        
        if 'best_params' in self.best_metrics:
            print(f"\nTham số tốt nhất:")
            for param, value in self.best_metrics['best_params'].items():
                print(f"  {param}: {value}")
    
    # ==================================================
    # HÀM VISUALIZATION
    # ==================================================
    def confusion_matrix_ROC(self):
        """
        Vẽ Confusion Matrix và ROC Curve cho model tốt nhất
        """
        if self.best_model is None:
            print("Chưa có model tốt nhất!")
            return
        
        y_pred = self.best_metrics["y_pred"]
        y_proba = self.best_metrics["y_proba"]
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=['Thực tế: Không bệnh', 'Thực tế: Có bệnh'],
            columns=['Dự đoán: Không bệnh', 'Dự đoán: Có bệnh']
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = self.best_metrics["roc_auc"]
        
        # Vẽ
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap Confusion Matrix
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Ma trận nhầm lẫn - {self.best_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dự đoán', fontsize=12)
        axes[0].set_ylabel('Thực tế', fontsize=12)
        
        # ROC Curve
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title(f'ROC Curve - {self.best_name}', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=10):
        """
        Vẽ biểu đồ feature importance
        """
        if self.feature_importance is None:
            print("Chưa có feature importance. Vui lòng train model trước.")
            return
        
        # Lấy top N features
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Độ quan trọng', fontsize=12)
        plt.title(f'Top {top_n} Đặc trưng Quan trọng nhất ({self.best_name})', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Thêm giá trị trên mỗi bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # In bảng feature importance
        print("\n" + "="*60)
        print("TOP ĐẶC TRƯNG QUAN TRỌNG NHẤT")
        print("="*60)
        print(top_features.to_string(index=False))
    
    def plot_learning_curve(self, cv=5):
        """
        Vẽ learning curve
        """
        if self.best_model is None:
            print("Chưa có model tốt nhất. Vui lòng train model trước.")
            return
        
        # Tính learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, self.X_train, self.y_train,
            cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Tính mean và std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Vẽ
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', 
                label='Training accuracy', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='green',
                label='Validation accuracy', linewidth=2)
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='green')
        
        plt.xlabel('Số lượng mẫu huấn luyện', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Learning Curve - {self.best_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ==================================================
    # HÀM LƯU KẾT QUẢ
    # ==================================================
    def save_comparison_results(self):
        """
        Lưu kết quả so sánh các model
        """
        results_path = os.path.join(self.save_dir, 'model_comparison_results.csv')
        
        # Tạo DataFrame từ all_results
        results_data = []
        for name, metrics in self.all_results.items():
            row = {'Model': name}
            for key, value in metrics.items():
                if key not in ['y_pred', 'y_proba']:
                    if key == 'best_params':
                        row[key] = str(value)
                    else:
                        row[key] = value
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_path, index=False)
        
        print(f"\nĐã lưu kết quả so sánh tại: {results_path}")
        return results_path
    
    # ==================================================
    # HÀM LƯU VÀ TẢI MODEL
    # ==================================================
    def save_model(self, model_name='best_GradientBoosting.pkl'):
        """
        Lưu mô hình tốt nhất và các thông tin liên quan
        """
        if self.best_model is None:
            print("Không có mô hình để lưu!")
            return None
        
        try:
            # Lưu model
            model_path = os.path.join(self.save_dir, model_name)
            joblib.dump(self.best_model, model_path)
            
            # Lưu metrics
            metrics_path = os.path.join(self.save_dir, model_name.replace(".pkl", "_metrics.pkl"))
            joblib.dump(self.best_metrics, metrics_path)
            
            # Lưu feature importance
            if self.feature_importance is not None:
                feature_path = os.path.join(self.save_dir, model_name.replace(".pkl", "_features.csv"))
                self.feature_importance.to_csv(feature_path, index=False)
            
            # Lưu toàn bộ kết quả
            all_results_path = os.path.join(self.save_dir, model_name.replace(".pkl", "_all_results.pkl"))
            joblib.dump({
                'all_results': self.all_results,
                'all_models_names': list(self.all_models.keys()),
                'best_model_name': self.best_name,
                'feature_names': self.feature_names,
                'dataset_info': {
                    'train_shape': self.X_train.shape,
                    'test_shape': self.X_test.shape,
                    'train_samples': len(self.y_train),
                    'test_samples': len(self.y_test)
                }
            }, all_results_path)
            
            print("\n" + "="*80)
            print("ĐÃ LƯU TẤT CẢ THÔNG TIN MODEL!")
            print("="*80)
            print(f" Model: {model_path}")
            print(f" Metrics: {metrics_path}")
            if self.feature_importance is not None:
                print(f"🔍 Feature importance: {feature_path}")
            print(f" All results: {all_results_path}")
            print(f" Save directory: {self.save_dir}")
            
            return model_path
            
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")
            return None
    
    def load_model(self, model_path):
        """
        Tải mô hình đã lưu
        """
        try:
            if os.path.exists(model_path):
                self.best_model = joblib.load(model_path)
                
                # Tải metrics nếu có
                metrics_path = model_path.replace(".pkl", "_metrics.pkl")
                if os.path.exists(metrics_path):
                    self.best_metrics = joblib.load(metrics_path)
                
                # Tải all results nếu có
                all_results_path = model_path.replace(".pkl", "_all_results.pkl")
                if os.path.exists(all_results_path):
                    all_data = joblib.load(all_results_path)
                    self.all_results = all_data.get('all_results', {})
                    self.best_name = all_data.get('best_model_name', 'Unknown')
                    self.feature_names = all_data.get('feature_names', [])
                
                print(f"\n Đã tải mô hình từ: {model_path}")
                print(f"   Tên model: {self.best_name}")
                print(f"   Loại model: {type(self.best_model).__name__}")
                
                return self.best_model
            else:
                print(f" Không tìm thấy file: {model_path}")
                return None
                
        except Exception as e:
            print(f" Lỗi khi tải model: {e}")
            return None
    
    # ==================================================
    # HÀM DỰ ĐOÁN
    # ==================================================
    def predict_new(self, X_new, return_proba=True, threshold=0.5):
        """
        Dự đoán trên dữ liệu mới
        
        Parameters:
        -----------
        X_new: Dữ liệu mới (đã chuẩn hóa)
        return_proba: Có trả về xác suất không
        threshold: Ngưỡng phân loại
        """
        if self.best_model is None:
            print("Vui lòng train hoặc load model trước!")
            return None
        
        try:
            # Đảm bảo X_new là numpy array
            if isinstance(X_new, pd.DataFrame):
                X_new = X_new.values
            
            # Dự đoán
            if return_proba:
                probabilities = self.best_model.predict_proba(X_new)
                predictions = (probabilities[:, 1] >= threshold).astype(int)
                
                # Tạo kết quả
                results = pd.DataFrame({
                    'prediction': predictions,
                    'probability_class_0': probabilities[:, 0],
                    'probability_class_1': probabilities[:, 1],
                    'prediction_label': np.where(predictions == 1, 'Có bệnh', 'Không bệnh')
                })
            else:
                predictions = self.best_model.predict(X_new)
                results = pd.DataFrame({
                    'prediction': predictions,
                    'prediction_label': np.where(predictions == 1, 'Có bệnh', 'Không bệnh')
                })
            
            print(f"\n Đã dự đoán {len(predictions)} mẫu")
            print(f"   Số mẫu 'Có bệnh': {sum(predictions == 1)}")
            print(f"   Số mẫu 'Không bệnh': {sum(predictions == 0)}")
            
            return results
            
        except Exception as e:
            print(f" Lỗi khi dự đoán: {e}")
            return None
    
    # ==================================================
    # HÀM IN TÓM TẮT
    # ==================================================
    def get_model_summary(self):
        """
        In tóm tắt thông tin model
        """
        if self.best_model is None:
            print(" Chưa có model tốt nhất.")
            return
        
        print("\n" + "="*80)
        print("TÓM TẮT MÔ HÌNH GRADIENT BOOSTING")
        print("="*80)
        print(f" Tên model: {self.best_name}")
        print(f" Loại model: {type(self.best_model).__name__}")
        
        # Thông tin model
        if hasattr(self.best_model, 'n_estimators'):
            print(f" Số cây: {self.best_model.n_estimators}")
        if hasattr(self.best_model, 'learning_rate'):
            print(f" Learning rate: {self.best_model.learning_rate}")
        if hasattr(self.best_model, 'max_depth'):
            print(f" Max depth: {self.best_model.max_depth}")
        
        print(f"\n Metrics trên tập test:")
        for key, value in self.best_metrics.items():
            if key not in ['y_pred', 'y_proba', 'best_params']:
                print(f"  {key}: {value:.4f}")
        
        print(f"\n Dataset info:")
        print(f"  Train shape: {self.X_train.shape}")
        print(f"  Test shape: {self.X_test.shape}")
        print(f"  Số đặc trưng: {len(self.feature_names)}")
        print(f"  Train samples: {len(self.y_train)}")
        print(f"  Test samples: {len(self.y_test)}")