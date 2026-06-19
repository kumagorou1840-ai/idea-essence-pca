import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="アイデア本質抽出(PCA)ダッシュボード", page_icon="🧠", layout="wide")

st.title("🧠 アイデア本質抽出(PCA)ダッシュボード")
st.markdown("##### バラバラな複数のアイデアの多次元評価から、主成分分析(PCA)を用いて「本質（最も重要な評価軸）」を抽出します。")

# --- セッション状態の初期化 ---
if "ideas" not in st.session_state:
    # 初期サンプルデータ
    st.session_state.ideas = [
        {"名前": "ぶっ飛んだ社会貢献", "Novelty": 0.9, "Feasibility": 0.2, "Social Impact": 0.8, "Excitement": 0.9, "Profitability": 0.3},
        {"名前": "堅実なビジネス", "Novelty": 0.3, "Feasibility": 0.9, "Social Impact": 0.2, "Excitement": 0.4, "Profitability": 0.8},
        {"名前": "バランス型", "Novelty": 0.7, "Feasibility": 0.5, "Social Impact": 0.6, "Excitement": 0.8, "Profitability": 0.5},
        {"名前": "純粋な情熱プロジェクト", "Novelty": 0.9, "Feasibility": 0.1, "Social Impact": 0.9, "Excitement": 0.9, "Profitability": 0.1},
        {"名前": "既存事業の改善", "Novelty": 0.2, "Feasibility": 0.8, "Social Impact": 0.3, "Excitement": 0.3, "Profitability": 0.9}
    ]

features = ["Novelty", "Feasibility", "Social Impact", "Excitement", "Profitability"]

# --- サイドバー：新規アイデアの追加 ---
st.sidebar.header("➕ 新しいアイデアの追加")
new_name = st.sidebar.text_input("アイデア名", placeholder="例：AI家庭教師アプリ")

new_values = {}
for f in features:
    new_values[f] = st.sidebar.slider(f"{f} (0.0〜1.0)", 0.0, 1.0, 0.5, step=0.1)

if st.sidebar.button("アイデアを追加する"):
    if new_name.strip() == "":
        st.sidebar.error("アイデア名を入力してください。")
    elif any(i["名前"] == new_name for i in st.session_state.ideas):
        st.sidebar.error("同名のアイデアが既に存在します。")
    else:
        new_idea = {"名前": new_name}
        new_idea.update(new_values)
        st.session_state.ideas.append(new_idea)
        st.sidebar.success(f"「{new_name}」を追加しました！")
        st.rerun()

# データのクリア・初期化
if st.sidebar.button("データを初期状態に戻す"):
    if "ideas" in st.session_state:
        del st.session_state.ideas
    st.rerun()

# --- メイン画面の処理 ---
df_ideas = pd.DataFrame(st.session_state.ideas)

# グリッドレイアウトでテーブル表示とPCA結果を表示
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("📋 登録されているアイデア一覧")
    
    # グラフ表示用のラベル（Idea 1, Idea 2...）を付与
    df_ideas.insert(0, "ラベル", [f"Idea {i+1}" for i in range(len(df_ideas))])
    
    st.dataframe(
        df_ideas[["ラベル", "名前"] + features], 
        use_container_width=True, 
        hide_index=True
    )

    # 削除機能
    st.write("---")
    st.markdown("##### 🗑️ アイデアの削除")
    delete_target = st.selectbox("削除するアイデアを選択", df_ideas["名前"].tolist())
    if st.button("選択したアイデアを削除"):
        st.session_state.ideas = [i for i in st.session_state.ideas if i["名前"] != delete_target]
        st.success(f"「{delete_target}」を削除しました。")
        st.rerun()

# PCA分析の実行
if len(df_ideas) >= 2:
    # 分析用データの抽出 (数値カラムのみ)
    X = df_ideas[features].values
    
    # PCAの実行 (第2主成分まで)
    n_comp = min(2, len(df_ideas))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    
    # 主成分（本質の軸）
    essence_axis = pca.components_[0]
    
    # 適合度スコアへの変換
    transformed = pca.transform(X)
    df_ideas["本質適合度 (PC1)"] = transformed[:, 0]
    if n_comp >= 2:
        df_ideas["副次的本質 (PC2)"] = transformed[:, 1]
    else:
        df_ideas["副次的本質 (PC2)"] = 0.0
        
    with col2:
        st.subheader("📊 アイデアの本質空間（PCA）")
        
        # 散布図の描画 (matplotlib)
        fig, ax = plt.subplots(figsize=(6, 5))
        x_vals = df_ideas["本質適合度 (PC1)"]
        y_vals = df_ideas["副次的本質 (PC2)"]
        
        ax.scatter(x_vals, y_vals, color='royalblue', s=150, edgecolors='black', alpha=0.8)
        
        # 各点に「Idea 1」「Idea 2」といった英語のラベルを付与（文字化け防止）
        for i, row in df_ideas.iterrows():
            ax.text(
                row["本質適合度 (PC1)"] + 0.02, 
                row["副次的本質 (PC2)"] + 0.02, 
                row["ラベル"], 
                fontsize=10, 
                fontweight='bold'
            )
            
        ax.set_title("Idea Distribution Space", fontsize=12)
        ax.set_xlabel("Primary Essence Score (PC1)", fontsize=10)
        ax.set_ylabel("Secondary Essence Score (PC2)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        
        st.pyplot(fig)

    # --- 本質（主成分）の解説エリア ---
    st.write("---")
    st.subheader("💡 抽出された『本質（評価軸）』の解説")
    
    # 寄与度の大きい特徴量を特定
    max_idx = np.argmax(np.abs(essence_axis))
    top_feature = features[max_idx]
    top_weight = essence_axis[max_idx]
    direction = "正（プラス）" if top_weight > 0 else "負（マイナス）"
    
    col_desc1, col_desc2 = st.columns([1, 1])
    
    with col_desc1:
        st.markdown(f"""
        現在のアイデア群から抽出された**「最も重要な本質の軸（第1主成分）」**において、
        一番影響力が大きいのは **{top_feature}**（影響度: `{top_weight:.3f}`）です。
        
        この分析から、登録されたアイデア群を評価する上で、
        **『{top_feature}』の{direction}の方向が最大の差別化要因・本質的な価値**になっていると言えます。
        """)
        
    with col_desc2:
        # ウェイトの可視化
        df_weights = pd.DataFrame({
            "特徴量": features,
            "本質への影響度(ウェイト)": essence_axis
        })
        st.dataframe(
            df_weights.sort_values("本質への影響度(ウェイト)", ascending=False), 
            use_container_width=True, 
            hide_index=True
        )

else:
    st.warning("PCA分析を行うには、アイデアが2つ以上登録されている必要があります。")
