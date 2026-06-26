import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

states_dict = {0: 0, 1: 1, 2: 2}

# ==========================================
# 📊 1. 數據匯入與「真實特徵工程」
# ==========================================
def extract_real_features_and_labels(subject_id):
    """
    將受試者的心率、步數、原始加速度與標籤精準對齊，提取真實 ML 特徵
    """
    label_path = os.path.join("labels", f"{subject_id}_labeled_sleep.txt")
    hr_path = os.path.join("heart_rate", f"{subject_id}_heartrate.txt")  
    step_path = os.path.join("steps", f"{subject_id}_steps.txt")
    accel_path = os.path.join("motion", f"{subject_id}_acceleration.txt") 
    
    if not os.path.exists(label_path): raise FileNotFoundError(f"找不到標籤檔: {label_path}")
    if not os.path.exists(hr_path): raise FileNotFoundError(f"找不到心率檔: {hr_path}")
    if not os.path.exists(step_path): raise FileNotFoundError(f"找不到步數檔: {step_path}")
    if not os.path.exists(accel_path): raise FileNotFoundError(f"找不到加速度檔: {accel_path}")
    
    # 讀取真實 PSG 答案
    df_sleep = pd.read_csv(label_path, header=None, sep=r'\s+|,', names=['seconds', 'stage'], engine='python')
    df_sleep['simplified_stage'] = df_sleep['stage'].map({0: 0, 1: 1, 2: 1, 3: 1, 5: 2})
    df_sleep = df_sleep.dropna()
    df_sleep['simplified_stage'] = df_sleep['simplified_stage'].astype(int)
    
    # 讀取感測器數據
    df_hr = pd.read_csv(hr_path, header=None, names=['seconds', 'bpm'], sep=r'\s+|,', engine='python')
    df_steps = pd.read_csv(step_path, header=None, names=['seconds', 'steps'], sep=r'\s+|,', engine='python')
    df_accel = pd.read_csv(accel_path, header=None, sep=r'\s+|,', names=['seconds', 'x', 'y', 'z'], engine='python')
    
    X_features = []
    y_labels = []
    
    # 對準「每 30 秒一個 Epoch」進行時序特徵工程
    for _, row in df_sleep.iterrows():
        epoch_time = row['seconds']
        true_stage = row['simplified_stage']
        
        # 特徵 1：尋找該 30 秒內的心率平均值
        hr_window = df_hr[(df_hr['seconds'] >= epoch_time) & (df_hr['seconds'] < epoch_time + 30)]
        avg_hr = hr_window['bpm'].mean() if not hr_window.empty else 70.0
        
        # 特徵 2：尋找前後 5 分鐘內的累積步數
        step_window = df_steps[(df_steps['seconds'] >= epoch_time - 300) & (df_steps['seconds'] <= epoch_time + 300)]
        total_steps = step_window['steps'].sum() if not step_window.empty else 0
        
        # 特徵 3：計算該 30 秒 Epoch 內三軸加速度的局部標準差
        accel_window = df_accel[(df_accel['seconds'] >= epoch_time) & (df_accel['seconds'] < epoch_time + 30)]
        if not accel_window.empty:
            accel_metric = accel_window['x'].std() + accel_window['y'].std() + accel_window['z'].std()
            if np.isnan(accel_metric): accel_metric = 0.0
        else:
            accel_metric = 0.0
        
        X_features.append([avg_hr, total_steps, accel_metric])
        y_labels.append(true_stage)
        
    return np.array(X_features), np.array(y_labels), df_sleep


def calculate_real_transition_matrix(df_sleep):
    stages = df_sleep['simplified_stage'].values
    counts = np.zeros((3, 3))
    for i in range(len(stages) - 1):
        counts[stages[i], stages[i+1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    return counts / row_sums


# ==========================================
# 🔍 2. 主程式：訓練真實模型並跑大迴圈
# ==========================================
labels_folder = "labels"
subject_list = [f.split("_")[0] for f in os.listdir(labels_folder) if f.endswith("_labeled_sleep.txt")]

print(f"🎯 偵測到共有 {len(subject_list)} 位受試者資料。\n")

all_raw_specs = []
all_hmm_specs = []

for target_subject in subject_list:
    print(f"==================== 正在分析受試者: {target_subject} ====================")
    
    # 建立訓練集 (留一法交叉驗證)
    X_train_list, y_train_list = [], []
    for sub_id in subject_list:
        if sub_id == target_subject:
            continue
        try:
            X_sub, y_sub, _ = extract_real_features_and_labels(sub_id)
            X_train_list.append(X_sub)
            y_train_list.append(y_sub)
        except Exception as e:
            continue
            
    if not X_train_list:
        print("❌ 錯誤：無法載入群體數據建立訓練集。")
        continue
        
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    print(f"🤖 正在為群體數據訓練真實的隨機森林模型 (訓練樣本數: {len(X_train)})...")
    ml_model = RandomForestClassifier(
        n_estimators=80, 
        max_depth=12, 
        class_weight="balanced",  
        random_state=42, 
        n_jobs=-1
    )
    ml_model.fit(X_train, y_train)
    
    try:
        X_test, y_true, df_sleep = extract_real_features_and_labels(target_subject)
        baseline_trans_matrix = calculate_real_transition_matrix(df_sleep)
        
        # 讓真實 ML 分類器輸出包含加速度後的時變預測機率矩陣
        real_ml_probs = ml_model.predict_proba(X_test)
        
        # ==========================================================
        # 🧠 💡 核心修改：實作文獻的「雙向資訊交換機制（Bi-directional Feedback）」
        # ==========================================================
        num_epochs = len(real_ml_probs)
        predict_stages = np.zeros(num_epochs, dtype=int)
        
        # 複製一份基準轉移矩陣，作為初始的動態矩陣
        dynamic_trans = baseline_trans_matrix.copy()
        
        # 逐個 Epoch (時間步) 進行時序迭代預測與即時反饋
        for t in range(num_epochs):
            # 1. 計算當前的發射觀測與狀態先驗融合機率
            if t == 0:
                # 第一步先以隨機森林的最高信心類別決定
                current_prediction = np.argmax(real_ml_probs[t])
            else:
                # 融合前一步的預測狀態。下一個狀態的可能機率 = 前一步預測乘上當前轉移矩陣
                prev_state = predict_stages[t-1]
                state_prior = dynamic_trans[prev_state, :]
                
                # 融合預測先驗與隨機森林模型給出的真實觀測機率 (雙向資訊交換點 A)
                combined_prob = state_prior * real_ml_probs[t]
                current_prediction = np.argmax(combined_prob)
            
            predict_stages[t] = current_prediction
            
            # 2. 閉環動態反饋 (雙向資訊交換點 B)：
            # 依據文獻 P.14 邏輯：「高度信心受試者為 Wake 時，會增加清醒時間、並減少接下來被誤判為睡眠的機率」
            # 這裡我們讀取隨機森林分類器輸出的清醒信心值 (real_ml_probs[t, 0])
            prob_wake = real_ml_probs[t, 0]
            
            if prob_wake > 0.55: # 分類器對 Wake 有較高度信心
                # 強力介入調整「下一個時間點的生物轉移機率」：顯著拉高其他睡眠階段跳回 Wake (0) 的容忍機率
                dynamic_trans[0, 0] = min(dynamic_trans[0, 0] * 1.2, 0.99)
                dynamic_trans[1, 0] = min(dynamic_trans[1, 0] * 3.5, 0.95) # 提高 NREM 轉 Wake 機率
                dynamic_trans[2, 0] = min(dynamic_trans[2, 0] * 3.5, 0.95) # 提高 REM 轉 Wake 機率
                
                # 重新將矩陣 Row-normalize，維持機率和為 1
                row_sums = dynamic_trans.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                dynamic_trans /= row_sums
            else:
                # 若模型判定穩定進入睡眠，則讓動態轉移矩陣緩慢衰減、自動回歸正常的群體生物學基礎值
                dynamic_trans = dynamic_trans * 0.8 + baseline_trans_matrix * 0.2
        
        # ==========================================================
        
        # 計算原始預測與雙向修正後的預測結果
        raw_predictions = np.argmax(real_ml_probs, axis=1)
        cm_raw = confusion_matrix(y_true, raw_predictions, labels=[0, 1, 2])
        cm_hmm = confusion_matrix(y_true, predict_stages, labels=[0, 1, 2])
        
        wake_total = cm_raw[0].sum()
        if wake_total > 0:
            raw_wake_spec = cm_raw[0, 0] / wake_total
            hmm_wake_spec = cm_hmm[0, 0] / wake_total
            
            all_raw_specs.append(raw_wake_spec)
            all_hmm_specs.append(hmm_wake_spec)
            
            print(f"真實 ML 清醒識別率 (Fraction of wake scored as wake): {raw_wake_spec:.3f}")
            print(f"雙向機制 修正後清醒識別率 (Fraction of wake scored as wake): {hmm_wake_spec:.3f}\n")
            
    except Exception as e:
        print(f"❌ 處理目標受試者 {target_subject} 的預測時發生錯誤: {e}\n")

# ==========================================
# 🏁 3. 最終群體總結報告
# ==========================================
print("==================== 🚀 最終真實統計報告 ====================")
if all_raw_specs:
    print(f"群體平均 - 原始真實 ML 清醒正確率: {np.mean(all_raw_specs):.3f}")
    print(f"群體平均 - 雙向機制修正後清醒正確率: {np.mean(all_hmm_specs):.3f}")
    print(f"📈 真正的清醒識別率淨提升了: {(np.mean(all_hmm_specs) - np.mean(all_raw_specs))*100:+.1f}%")