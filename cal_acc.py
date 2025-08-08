import json
import pandas as pd

def calculate_prediction_accuracy(json_path, csv_path):
    """
    计算模型预测结果与真实标签之间的准确率。
    - 整体准确率
    - 每个特征的独立准确率
    - 每个病人的独立准确率【并以正确/错误列表形式显示】
    """
    # --- 第1步和第2步保持不变 ---
    try:
        with open(json_path, 'r') as f:
            predictions_data = json.load(f)
        label_df = pd.read_csv(csv_path, dtype={'hospital number': str})
        print("成功加载预测文件和标签文件。")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请检查文件路径是否正确。")
        return

    pattern_keys = [f"pattern_{i+1}" for i in range(9)]
    # feat_columns = ['feat36', 'feat41', 'feat43', 'feat47', 'feat52', 'feat57', 'feat59', 'feat61', 'feat62', 'feat64', 'feat68']
    feat_columns = ['feat36', 'feat41', 'feat43', 'feat47', 'feat52', 'feat57', 'feat59', 'feat61', 'feat64']
    feature_map = dict(zip(pattern_keys, feat_columns))
    
    # --- 第3步：初始化计数器 ---
    total_comparisons, correct_predictions = 0, 0
    per_feature_counts = {key: {'correct': 0, 'total': 0} for key in pattern_keys}
    per_patient_results = {}

    # --- 第4步：遍历、对比并计算 ---
    print("开始逐一对比预测结果与真实标签...")
    for patient_id, patient_data in predictions_data.items():
        # 检查数据格式是否正确，并提取真正的预测结果
        if not isinstance(patient_data, dict) or 'final_answers' not in patient_data:
            print(f"警告：病人 {patient_id} 的数据格式不正确或缺少 'final_answers'，跳过。")
            continue
        
        predictions = patient_data['final_answers']

        true_labels_row = label_df[label_df['hospital number'] == patient_id]
        if true_labels_row.empty:
            print(f"警告：在 label_df.csv 中找不到病人ID为 {patient_id} 的记录，跳过。")
            continue

        patient_correct_count, patient_total_count = 0, 0
        
        # --- (新增代码：为当前病人创建正确/错误列表) ---
        correct_patterns = []
        incorrect_patterns = []

        # 遍历11个特征进行对比
        for pattern_key, feat_col in feature_map.items():
            if pattern_key not in predictions:
                continue

            model_prediction = predictions[pattern_key]
            true_label = true_labels_row[feat_col].iloc[0]

            pattern_number = pattern_key.replace('pattern_', '')
            
            try:
                is_correct = int(model_prediction) == int(true_label)
                if is_correct:
                    correct_predictions += 1
                    per_feature_counts[pattern_key]['correct'] += 1
                    patient_correct_count += 1
                    correct_patterns.append(pattern_number) 
                else:
                    incorrect_patterns.append(pattern_number) 
            except (ValueError, TypeError):
                continue

            total_comparisons += 1
            per_feature_counts[pattern_key]['total'] += 1
            patient_total_count += 1
            
        # --- (修改代码：将新的列表存入结果) ---
        if patient_total_count > 0:
            patient_accuracy = (patient_correct_count / patient_total_count) * 100
            per_patient_results[patient_id] = {
                'accuracy': f"{patient_accuracy:.2f}%",
                'summary': f"({patient_correct_count}/{patient_total_count})",
                'correct_list': correct_patterns,    # <--- 新增
                'incorrect_list': incorrect_patterns # <--- 新增
            }

    # --- 第5步：计算并打印最终结果 ---
    print("\n--- 准确率计算完成 ---")
    if total_comparisons > 0:
        # (打印整体和各特征准确率的代码保持不变)
        overall_accuracy = (correct_predictions / total_comparisons) * 100
        print(f"整体准确率 (Overall Accuracy): {overall_accuracy:.2f}% ({correct_predictions}/{total_comparisons})")
        
        print("\n--- 各个特征的独立准确率 ---")
        for pattern_key, counts in per_feature_counts.items():
            if counts['total'] > 0:
                feat_col = feature_map[pattern_key]
                feature_accuracy = (counts['correct'] / counts['total']) * 100
                print(f"{pattern_key} ({feat_col}): {feature_accuracy:.2f}% ({counts['correct']}/{counts['total']})")

        # --- (打印每个病人的准确率) ---
        print("\n--- 每个病人的独立准确率 ---")
        for patient_id, results in per_patient_results.items():
            print(f"病人ID {patient_id}: {results['accuracy']} {results['summary']}")

        # --- (修改代码：以新的、简洁的格式打印每个病人的结果) ---
        print("\n--- 每个病人的独立准确率及详情 ---")
        for patient_id, results in per_patient_results.items():
            print(f"\n病人ID {patient_id}: {results['accuracy']} {results['summary']}")
            # 打印正确和错误的列表
            print(f" - Correct: {results['correct_list']}")
            print(f" - Incorrect: {results['incorrect_list']}")

    else:
        print("未能进行任何有效的对比。")

# --- 主程序入口 ---
if __name__ == "__main__":
    json_file_path = 'agent_results_20_reasoning.json'
    csv_file_path = 'label_df.csv'
    
    calculate_prediction_accuracy(json_file_path, csv_file_path)
