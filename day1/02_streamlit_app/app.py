import streamlit as st
import ui
import llm
import database
import metrics
import data
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 簡易的なカスタムCSS ---
st.markdown("""
<style>
    /* チャットメッセージのスタイル */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
        display: flex;
    }
    .user-message {
        background-color: #F0F4F9;
        margin-left: auto;
        max-width: 80%;
    }
    .ai-message {
        background-color: #6C63FF;
        color: white;
        margin-right: auto;
        max-width: 80%;
    }
    /* ボタンのスタイル */
    .stButton>button {
        border-radius: 0.5rem;
    }
    /* タイトルのスタイル */
    h1 {
        color: #6C63FF;
    }
</style>
""", unsafe_allow_html=True)

# --- 初期化処理 ---
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()

# --- モデルのロード ---
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"使用デバイス: {device}")
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。")
        return None

pipe = llm.load_model()

# --- タイトルと説明 ---
st.title("🤖 Gemma チャットボット")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックができます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = "チャット"  # デフォルトページ

# ページ選択
page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page),
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector)
)

# モデル情報の表示
st.sidebar.markdown("---")
st.sidebar.subheader("モデル情報")
st.sidebar.markdown(f"**モデル名**: {MODEL_NAME}")
st.sidebar.markdown(f"**実行環境**: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        # 注意！引数の順序が正しいことを確認 (修正ポイント)
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    # 履歴ページを表示する前にデータの有無をチェックするようui.pyを修正することを想定
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッター ---
st.sidebar.markdown("---")
st.sidebar.info("Gemmaは、Google DeepMindによって開発されたオープンウェイトの言語モデルです。")