import pandas as pd
from Loaddata import load_data
from eda1 import EDA
from preprocessing import Preprocessor
from MLP_NNCGIANG import MLPModel
from evalution import ModelEvaluator
from model_gradient_boosting_NguyenMinhThien import GradientBoostingModel
from DecisionTree_NTYNgoc import DecisionTreePipeline 
import joblib
def main():
    # 1. Load dữ liệu
    df = load_data()

    # 2. Tiền xử lý
    pre = Preprocessor(df)
    
    result = pre.run()
    # Lưu preprocessor
    joblib.dump(pre, "models/preprocessor.pkl")
    #3. Khởi tạo các đối tượng EDA
    
    eda = EDA(df)

    # Gọi các phương thức
    eda.eda_overview()
    eda.missingvalue()
    eda.outlier() #Biểu đồ phân tích ngoại lai
    eda.target_anal() #Biểu đồ phân dặc trưng
    eda.heartdisease_anal() #Biểu đồ phân tích biến mục tiêu
    eda.sex_anal() #Biểu đồ phân tích giới tính theo biến mục tiêu
    eda.age_anal() #Biểu đồ phân tích độ tuổi theo biến mục tiêu
    eda.heatmap(df) #Ma trận tương quan
    
    if result is not None:
        #Lưu các dữ liệu
        X_train_scaled, X_test_scaled, y_train, y_test = result
        
        print("=====MLPClassifier=====")
        #=======MLPClassifier=====
        #Training model MLPClassifier
        trainer = MLPModel(X_train_scaled, X_test_scaled, y_train, y_test)
        #gọi hàm run để chạy model
        best_model, best_name, best_metrics = trainer.run()
        #In model cao nhất
        print("\nMô hình cuối cùng được chọn:", best_name)
        print("Accuracy:", best_metrics["accuracy"])
        #Lưu model MLP
        trainer.save_model('best_MLP.pkl')

        #=======Decision Tree=====
        dt_pipeline = DecisionTreePipeline(X_train_scaled, X_test_scaled, 
                                           y_train, y_test, 
                                           feature_names=list(pre.X.columns))

        # Train cơ bản
        model = dt_pipeline.train_decision_tree()
        metrics = dt_pipeline.evaluate_model(model)
        print(metrics)

        # Vẽ confusion matrix
        dt_pipeline.plot_confusion_matrix(metrics["y_pred"])

        # Vẽ cây
        dt_pipeline.plot_decision_tree(model)

        # Tune max_depth
        best_depth, best_score = dt_pipeline.tune_max_depth()

        # Grid search
        best_dt = dt_pipeline.grid_search_decision_tree()

        # Accuracy vs Depth
        dt_pipeline.plot_accuracy_vs_depth()

        # ROC Curve
        dt_pipeline.plot_roc_curve(metrics["y_proba"])
        #Lưu model
        dt_pipeline.save_model('best_DecisionTree.pkl')
        
        #======GradientBoostingModel=======
        gbM = GradientBoostingModel(X_train_scaled, X_test_scaled, 
                                           y_train, y_test)
        
        # Huấn luyện mô hình
        models = gbM.run_models()
        
        #In ra mô hình tốt nhất
        print(f"Mô hình tốt nhất: {gbM.best_model}")
        #Lưu model
        gbM.save_model('best_GradientBoosting.pkl')
        
        #Lưu dict để so sánh
        results = {
            "MLP": best_metrics,
            "DecisionTree": metrics,
            "GradientBoosting": gbM.best_metrics
        }

        evaluator = ModelEvaluator(y_test, results)

        # Vẽ biểu đồ so sánh
        evaluator.plot_bar_metrics()
        evaluator.plot_heatmap()
        evaluator.plot_roc_all_models()

        # In mô hình tốt nhất theo accuracy
        evaluator.best_model("accuracy")
    
if __name__ == "__main__":
    main()
