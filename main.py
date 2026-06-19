import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_idea_essence(ideas_data, feature_names):
    """
    アイデアの多次元データから「本質（主成分）」を抽出する
    """
    print("--- アイデア本質抽出シミュレーション ---")
    
    # データの正規化（平均を0に）
    X = np.array(ideas_data)
    
    # PCAの実行（第1主成分から第2主成分まで）
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # 主成分（本質の軸）の確認
    essence_axis = pca.components_[0]
    print("\n[抽出された『本質』の構成要素]")
    for name, weight in zip(feature_names, essence_axis):
        print(f"{name}: {weight:.3f}")
        
    # 各アイデアの「本質スコア」への変換
    transformed_ideas = pca.transform(X)
    
    print("\n[各アイデアの本質への適合度（第1主成分スコア）]")
    for i, score in enumerate(transformed_ideas[:, 0]):
        print(f"アイデア {i+1}: {score:.3f}")
        
    return essence_axis, transformed_ideas

if __name__ == "__main__":
    # 特徴量：[新規性, 実現性, 社会的意義, 個人的ワクワク度, 収益性]
    features = ["Novelty", "Feasibility", "Social Impact", "Excitement", "Profitability"]
    
    # サンプルデータ（5つの断片的なアイデア）
    raw_ideas = [
        [0.9, 0.2, 0.8, 0.9, 0.3], # ぶっ飛んだ社会貢献
        [0.3, 0.9, 0.2, 0.4, 0.8], # 堅実なビジネス
        [0.7, 0.5, 0.6, 0.8, 0.5], # バランス型
        [0.9, 0.1, 0.9, 0.9, 0.1], # 純粋な情熱プロジェクト
        [0.2, 0.8, 0.3, 0.3, 0.9]  # 既存事業の改善
    ]
    
def plot_ideas(transformed_ideas, output_path):
    """
    抽出された主成分空間（PC1, PC2）上に各アイデアをプロットし、画像として保存する
    """
    plt.figure(figsize=(8, 6))
    
    # 散布図の描画
    x = transformed_ideas[:, 0]
    y = transformed_ideas[:, 1]
    plt.scatter(x, y, color='royalblue', s=100, edgecolors='black', alpha=0.8)
    
    # 各点にラベル（アイデア 1, 2, ...）を付与
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi + 0.02, yi + 0.02, f"Idea {i+1}", fontsize=12)
    
    # 軸ラベルとタイトルの設定（日本語フォントがない場合に備え英語で記述）
    plt.title("Idea Distribution in Essence Space (PCA)", fontsize=14)
    plt.xlabel("Primary Essence Score (PC1)", fontsize=12)
    plt.ylabel("Secondary Essence Score (PC2)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    
    # 画像ファイルとして保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Chart] Saved chart to: {output_path}")

if __name__ == "__main__":
    # 特徴量：[新規性, 実現性, 社会的意義, 個個人的ワクワク度, 収益性]
    features = ["Novelty", "Feasibility", "Social Impact", "Excitement", "Profitability"]
    
    # サンプルデータ（5つの断片的なアイデア）
    raw_ideas = [
        [0.9, 0.2, 0.8, 0.9, 0.3], # ぶっ飛んだ社会貢献
        [0.3, 0.9, 0.2, 0.4, 0.8], # 堅実なビジネス
        [0.7, 0.5, 0.6, 0.8, 0.5], # バランス型
        [0.9, 0.1, 0.9, 0.9, 0.1], # 純粋な情熱プロジェクト
        [0.2, 0.8, 0.3, 0.3, 0.9]  # 既存事業の改善
    ]
    
    essence, scores = analyze_idea_essence(raw_ideas, features)
    
    # グラフのプロットと保存
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "idea_essence_chart.png")
    plot_ideas(scores, chart_path)
    
    print("\n[OK] 数式化完了：バラバラなアイデアから『最も重要な軸』が抽出されました。")
