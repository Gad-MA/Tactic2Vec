import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SCENES_PATH = 'aligned_scenes.pkl'
VECTORS_PATH = 'vectors.npy'
OUTPUT_FILE = 'embedding_space.png'

def get_rotating_html(fig, plot_id):
    """Returns HTML for a figure with auto-rotation JS."""
    # div_id = f"plot_{plot_id}"
    js_rotation = f"""
    var gd = document.getElementById('{plot_id}');
    var theta = 0;
    function animate() {{
        theta += 0.005;
        Plotly.relayout(gd, {{
            'scene.camera.eye': {{
                x: 1.8 * Math.cos(theta),
                y: 1.8 * Math.sin(theta),
                z: 1.0
            }}
        }});
        requestAnimationFrame(animate);
    }}
    animate();
    """
    return fig.to_html(full_html=False, include_plotlyjs='cdn', div_id=plot_id, post_script=js_rotation)

def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive Matplotlib backend")
    parser.add_argument("--plotly", action="store_true", help="Generate interactive Plotly visualization (HTML)")
    args = parser.parse_args()

    if args.interactive:
        print("Enabling interactive Matplotlib backend (TkAgg)...")
        matplotlib.use('TkAgg')

    if not os.path.exists(SCENES_PATH) or not os.path.exists(VECTORS_PATH):
        print("Scenes or Vectors not found.")
        return

    # 1. Load Data
    print(f"Loading vectors from {VECTORS_PATH}...")
    vectors = np.load(VECTORS_PATH)
    print(f"Vectors shape: {vectors.shape}")

    print(f"Loading metadata from {SCENES_PATH}...")
    with open(SCENES_PATH, 'rb') as f:
        scenes = pickle.load(f)

    # Extract labels
    labels = []
    for s in scenes:
        meta = s.get('event_metadata', {})
        label = meta.get('type', 'Unknown')
        labels.append(label)
    
    unique_labels = list(set(labels))
    label_to_color = {l: i for i, l in enumerate(unique_labels)}
    c = [label_to_color[l] for l in labels]

    # 2. Dimensionality Reduction
    print("Computing PCA (2D)...")
    pca = PCA(n_components=2)
    vec_pca = pca.fit_transform(vectors)

    print("Computing t-SNE (2D)...")
    # Perplexity should be less than number of samples
    metrics_perp = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=metrics_perp, random_state=42)
    vec_tsne = tsne.fit_transform(vectors)

    # 3. Plotting
    print(f"Generating plot: {OUTPUT_FILE}...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA Plot
    sc1 = axes[0].scatter(vec_pca[:, 0], vec_pca[:, 1], c=c, cmap='tab10', alpha=0.7)
    axes[0].set_title(f'PCA Projection (N={len(vectors)})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # Legend for PCA
    handles, _ = sc1.legend_elements(prop='colors')
    axes[0].legend(handles, unique_labels, title="Event Type", loc="upper right")

    # t-SNE Plot
    sc2 = axes[1].scatter(vec_tsne[:, 0], vec_tsne[:, 1], c=c, cmap='tab10', alpha=0.7)
    axes[1].set_title(f't-SNE Projection (Perp={metrics_perp})')
    axes[1].set_xlabel('Dim 1')
    axes[1].set_ylabel('Dim 2')
    

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"Saved 2D plot to {OUTPUT_FILE}")
    
    # 4. 3D Plotting
    print("Computing PCA (3D)...")
    pca_3d = PCA(n_components=3)
    vec_pca_3d = pca_3d.fit_transform(vectors)
    
    print("Computing t-SNE (3D)...")
    tsne_3d = TSNE(n_components=3, perplexity=metrics_perp, random_state=42)
    vec_tsne_3d = tsne_3d.fit_transform(vectors)
    
    print("Generating 3D plots...")
    fig_3d = plt.figure(figsize=(16, 8))
    
    # PCA 3D
    ax1 = fig_3d.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(vec_pca_3d[:, 0], vec_pca_3d[:, 1], vec_pca_3d[:, 2], c=c, cmap='tab10', alpha=0.8)
    ax1.set_title('PCA 3D')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    
    # t-SNE 3D
    ax2 = fig_3d.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(vec_tsne_3d[:, 0], vec_tsne_3d[:, 1], vec_tsne_3d[:, 2], c=c, cmap='tab10', alpha=0.8)
    ax2.set_title('t-SNE 3D')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Dim 3')
    
    # Legend
    handles, _ = sc1.legend_elements(prop='colors')
    fig_3d.legend(handles, unique_labels, title="Event Type", loc="upper right")
    
    output_3d = 'embedding_space_3d.png'
    plt.savefig(output_3d, dpi=150)
    print(f"Saved 3D plot to {output_3d}")

    # 5. Plotly Interactive
    if args.plotly:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            print("Generating interactive Plotly visualization...")

            import pandas as pd
            df_pca = pd.DataFrame(vec_pca_3d, columns=['PC1', 'PC2', 'PC3'])
            df_pca['Event Type'] = labels
            
            fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3',
                                    color='Event Type', title='PCA 3D Interactive',
                                    height=800)
            fig_pca.update_traces(marker=dict(size=3))
            fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=50))
            
            df_tsne = pd.DataFrame(vec_tsne_3d, columns=['x', 'y', 'z'])
            df_tsne['Event Type'] = labels
            
            fig_tsne = px.scatter_3d(df_tsne, x='x', y='y', z='z',
                                     color='Event Type', title='t-SNE 3D Interactive',
                                     height=800)
            fig_tsne.update_traces(marker=dict(size=3))
            fig_tsne.update_layout(margin=dict(l=0, r=0, b=0, t=50))
            
            html_out = 'embedding_space_interactive.html'
            with open(html_out, 'w') as f:
                f.write("<html><head><title>Embedding Space Interactive</title></head><body>")
                f.write(get_rotating_html(fig_pca, "pca_3d"))
                f.write("<hr>")
                f.write(get_rotating_html(fig_tsne, "tsne_3d"))
                f.write("</body></html>")
            
            print(f"Saved Plotly interactive visualization to {html_out}")
        except ImportError:
            print("Plotly or Pandas not installed. Run 'pip install plotly pandas' to enable Plotly visualization.")

    if args.interactive:
        print("Showing interactive plots. Close windows to exit.")
        plt.show()

if __name__ == "__main__":
    main()
