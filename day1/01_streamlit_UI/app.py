import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import japanize_matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# ============================================
# ページ設定 - 必ず最初に呼び出す
# ============================================
st.set_page_config(
    page_title="データサイエンスダッシュボード",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 日本語フォント設定
# ============================================
# 日本語フォント設定（システムにインストールされているフォントを探す）
japanese_fonts = ['IPAGothic', 'Meiryo', 'Yu Gothic', 'MS Gothic', 'Hiragino Sans']
font_found = False

for font in japanese_fonts:
    if any([f for f in fm.fontManager.ttflist if font in f.name]):
        plt.rcParams['font.family'] = font
        font_found = True
        break

if not font_found:
    # システムに日本語フォントがない場合の対処
    st.warning("日本語フォントが見つかりませんでした。一部の文字が正しく表示されない可能性があります。")

# グラフの設定
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
plt.rcParams['font.size'] = 12

# カスタムCSS
st.markdown("""
<style>
    body {
        font-family: 'Meiryo', 'Yu Gothic', 'Hiragino Sans', sans-serif;
    }
    .main-title {
        color: #1E88E5;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
        border-bottom: 2px solid #1E88E5;
    }
    .sub-title {
        color: #26A69A;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 5px solid #26A69A;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        color: #616161;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #1976D2;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-delta {
        font-size: 14px;
    }
    .metric-good {
        color: #4CAF50;
    }
    .metric-bad {
        color: #F44336;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #78909C;
        font-size: 12px;
        margin-top: 50px;
        border-top: 1px solid #ECEFF1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# サイドバー 
# ============================================
st.sidebar.markdown("# 分析設定")

# データセット選択
dataset_name = st.sidebar.selectbox(
    "データセット",
    ["Iris", "Wine", "Breast Cancer"]
)

# データセット名の日本語対応
dataset_name_ja = {
    "Iris": "アヤメ",
    "Wine": "ワイン",
    "Breast Cancer": "乳がん"
}

# モデル選択
model_name = st.sidebar.selectbox(
    "機械学習モデル",
    ["ランダムフォレスト", "ロジスティック回帰", "サポートベクターマシン"]
)

# モデルのハイパーパラメータ設定
st.sidebar.markdown("## モデルパラメータ")

if model_name == "ランダムフォレスト":
    n_estimators = st.sidebar.slider("木の数", 10, 100, 50, 5)
    max_depth = st.sidebar.slider("最大深さ", 2, 20, 10, 1)
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42
    }
elif model_name == "ロジスティック回帰":
    C = st.sidebar.slider("正則化パラメータ", 0.01, 10.0, 1.0, 0.01)
    max_iter = st.sidebar.slider("最大反復回数", 100, 1000, 500, 50)
    model_params = {
        "C": C,
        "max_iter": max_iter,
        "random_state": 42
    }
else:  # "サポートベクターマシン"
    C = st.sidebar.slider("正則化パラメータ", 0.01, 10.0, 1.0, 0.01)
    kernel = st.sidebar.selectbox("カーネル", ["linear", "rbf", "poly"])
    model_params = {
        "C": C,
        "kernel": kernel,
        "random_state": 42
    }

# トレーニングパラメータ
st.sidebar.markdown("## トレーニングパラメータ")
test_size = st.sidebar.slider("テストデータ割合", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.slider("ランダムシード", 1, 100, 42, 1)

# ============================================
# データロードと前処理
# ============================================
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [data.target_names[t] for t in data.target]
        # 日本語特徴量名に変換
        feature_names_ja = ['がく片長さ', 'がく片幅', '花びら長さ', '花びら幅']
        feature_names_mapping = dict(zip(data.feature_names, feature_names_ja))
        df = df.rename(columns=feature_names_mapping)
        # 日本語クラス名
        target_names_ja = ['セトサ', 'バーシコロル', 'バージニカ']
        target_names_mapping = dict(zip(range(len(data.target_names)), target_names_ja))
        return df, feature_names_ja, target_names_ja
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [f"クラス {t}" for t in data.target]
        return df, data.feature_names, [f"クラス {i}" for i in range(len(data.target_names))]
    else:  # "Breast Cancer"
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [data.target_names[t] for t in data.target]
        # 日本語クラス名
        target_names_ja = ['悪性', '良性']
        return df, data.feature_names, target_names_ja

# データの読み込み
df, feature_names, target_names = load_data(dataset_name)

# ============================================
# モデルの訓練と評価
# ============================================
@st.cache_data
def train_and_evaluate_model(model_name, model_params, X, y, test_size, random_state):
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデルの選択
    if model_name == "ランダムフォレスト":
        model = RandomForestClassifier(**model_params)
    elif model_name == "ロジスティック回帰":
        model = LogisticRegression(**model_params)
    else:  # "サポートベクターマシン"
        model = SVC(**model_params, probability=True)
    
    # 訓練開始時間
    start_time = time.time()
    
    # モデルの訓練
    model.fit(X_train_scaled, y_train)
    
    # 訓練終了時間
    training_time = time.time() - start_time
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    
    # 各種メトリクスの計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    
    # 特徴量重要度（ランダムフォレストの場合のみ）
    if model_name == "ランダムフォレスト":
        feature_importance = model.feature_importances_
    else:
        feature_importance = None
    
    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "training_time": training_time,
        "feature_importance": feature_importance,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "scaler": scaler
    }

# モデルの訓練と評価
X = df[feature_names].values
y = df['target'].values
results = train_and_evaluate_model(model_name, model_params, X, y, test_size, random_state)

# ============================================
# メインコンテンツ
# ============================================
st.markdown('<div class="main-title">データサイエンスダッシュボード</div>', unsafe_allow_html=True)

# データセット情報
st.markdown('<div class="sub-title">データセット情報</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {dataset_name_ja[dataset_name]}データセット")
    st.write(f"**サンプル数:** {df.shape[0]}")
    st.write(f"**特徴量数:** {len(feature_names)}")
    st.write(f"**クラス数:** {len(target_names)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### クラス分布")
    class_counts = df['target_name'].value_counts()
    
    # 円グラフ
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(class_counts, autopct='%1.1f%%')
    # クラス名を日本語で設定
    ax.legend(class_counts.index, loc="best")
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 基本統計量")
    st.write(df[feature_names].describe().round(2))
    st.markdown('</div>', unsafe_allow_html=True)

# データ表示
st.markdown('<div class="sub-title">データプレビュー</div>', unsafe_allow_html=True)
with st.expander("データを表示", expanded=False):
    st.dataframe(df.drop(columns=['target']), use_container_width=True)

# モデル情報と評価指標
st.markdown('<div class="sub-title">モデル評価</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">正解率 (Accuracy)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{results["accuracy"]:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">適合率 (Precision)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{results["precision"]:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">再現率 (Recall)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{results["recall"]:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">F1スコア</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{results["f1"]:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 混同行列と特徴量重要度
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 混同行列")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = results["confusion_matrix"]
    
    # 表示のためのラベル取得
    if len(target_names) <= 10:  # ラベルが多すぎる場合は数値で表示
        labels = target_names
    else:
        labels = [str(i) for i in range(len(target_names))]
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='正解ラベル',
           xlabel='予測ラベル')
    
    # 値を表示
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if results["feature_importance"] is not None:
        st.markdown("### 特徴量重要度")
        importance = results["feature_importance"]
        indices = np.argsort(importance)[-10:]  # 上位10個の特徴量
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(range(len(indices)), importance[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('相対的重要度')
        ax.set_title('特徴量重要度（上位10件）')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.markdown("### モデル情報")
        st.write(f"**モデル:** {model_name}")
        st.write(f"**パラメータ:**")
        for param, value in model_params.items():
            st.write(f"- {param}: {value}")
        st.write(f"**訓練時間:** {results['training_time']:.4f} 秒")
    st.markdown('</div>', unsafe_allow_html=True)

# データ可視化
st.markdown('<div class="sub-title">データ可視化</div>', unsafe_allow_html=True)

# 特徴量の選択（散布図用）
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X軸の特徴量", feature_names)
with col2:
    y_axis = st.selectbox("Y軸の特徴量", feature_names, index=1)

# 散布図
fig, ax = plt.subplots(figsize=(10, 6))
for i, target in enumerate(np.unique(df['target'])):
    target_name = target_names[i] if i < len(target_names) else f"クラス {i}"
    mask = df['target'] == target
    ax.scatter(df[mask][x_axis], df[mask][y_axis], label=target_name, alpha=0.7)

ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title(f'{x_axis} vs {y_axis} (クラス別)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# 予測結果の可視化
st.markdown('<div class="sub-title">予測結果</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["予測結果の可視化", "誤分類サンプル"])

with tab1:
    # テストデータでの予測結果の可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # テストデータのインデックスを取得
    indices = np.arange(len(results["y_test"]))
    
    # 正解と予測を可視化
    ax.scatter(indices, results["y_test"], label="正解", alpha=0.7, marker='o')
    ax.scatter(indices, results["y_pred"], label="予測", alpha=0.7, marker='x')
    
    # 正解と予測が一致しないポイントをハイライト
    mismatch = results["y_test"] != results["y_pred"]
    if np.any(mismatch):
        ax.scatter(indices[mismatch], results["y_pred"][mismatch], 
                color='red', marker='x', s=100, label="誤分類")
    
    ax.set_xlabel("サンプルインデックス")
    ax.set_ylabel("クラス")
    ax.set_title("テストデータでの予測結果")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Y軸のラベルを調整
    if len(target_names) <= 10:
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_yticklabels(target_names)
    
    st.pyplot(fig)

with tab2:
    # 誤分類されたサンプルの詳細
    mismatch = results["y_test"] != results["y_pred"]
    if np.any(mismatch):
        st.markdown("### 誤分類サンプル一覧")
        
        # 元のスケールに戻す
        X_test_original = results["scaler"].inverse_transform(results["X_test"])
        
        # 誤分類データを取得
        X_misclassified = X_test_original[mismatch]
        y_test_misclassified = results["y_test"][mismatch]
        y_pred_misclassified = results["y_pred"][mismatch]
        
        # データフレームに変換
        misclassified_df = pd.DataFrame(X_misclassified, columns=feature_names)
        
        # 正解クラスと予測クラスの日本語名を追加
        misclassified_df['正解クラス'] = [target_names[int(y)] if int(y) < len(target_names) else f"クラス {int(y)}" 
                                     for y in y_test_misclassified]
        misclassified_df['予測クラス'] = [target_names[int(y)] if int(y) < len(target_names) else f"クラス {int(y)}" 
                                     for y in y_pred_misclassified]
        
        st.dataframe(misclassified_df, use_container_width=True)
        st.write(f"誤分類サンプル数: {np.sum(mismatch)} / {len(results['y_test'])} ({np.sum(mismatch)/len(results['y_test'])*100:.2f}%)")
    else:
        st.success("すべてのテストサンプルが正しく分類されました！")

# インタラクティブな予測
st.markdown('<div class="sub-title">インタラクティブ予測</div>', unsafe_allow_html=True)

with st.expander("特徴量を入力して予測"):
    cols = st.columns(4)
    user_input = {}
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 4
        with cols[col_idx]:
            # データセットの最小値と最大値を取得
            min_val = df[feature].min()
            max_val = df[feature].max()
            mean_val = df[feature].mean()
            
            # スライダーで入力
            user_input[feature] = st.slider(
                feature, 
                float(min_val), 
                float(max_val), 
                float(mean_val),
                step=float((max_val - min_val) / 100)
            )
    
    # 予測ボタン
    if st.button("予測する"):
        # 入力をスケーリング
        user_input_df = pd.DataFrame([user_input])
        user_input_scaled = results["scaler"].transform(user_input_df)
        
        # 予測
        prediction = results["model"].predict(user_input_scaled)[0]
        prediction_proba = None
        
        # 確率を取得（可能な場合）
        if hasattr(results["model"], "predict_proba"):
            prediction_proba = results["model"].predict_proba(user_input_scaled)[0]
        
        # 結果表示
        st.success(f"予測クラス: {target_names[int(prediction)] if int(prediction) < len(target_names) else f'クラス {int(prediction)}'}")
        
        # 確率表示（可能な場合）
        if prediction_proba is not None:
            st.write("クラス確率:")
            proba_df = pd.DataFrame({
                'クラス': [target_names[i] if i < len(target_names) else f'クラス {i}' for i in range(len(prediction_proba))],
                '確率': prediction_proba
            })
            proba_df = proba_df.sort_values('確率', ascending=False)
            
            # 棒グラフで表示
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(proba_df['クラス'], proba_df['確率'])
            ax.set_xlabel('確率')
            ax.set_ylabel('クラス')
            ax.set_title('予測確率')
            st.pyplot(fig)

# フッター
st.markdown("""
<div class="footer">
    <p>© 2025 データサイエンスダッシュボード | scikit-learn と Streamlit で作成</p>
    <p>このアプリケーションは機械学習と可視化の学習のために作成されました。</p>
</div>
""", unsafe_allow_html=True)
