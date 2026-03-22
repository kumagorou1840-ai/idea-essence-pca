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
    
    essence, scores = analyze_idea_essence(raw_ideas, features)
    
    print("\n✅ 数式化完了：バラバラなアイデアから『最も重要な軸』が抽出されました。")
