import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

def load_dataframes():
    """Loads the Parquet files and returns a dictionary of DataFrames."""
    df_hdbscan_1 = pd.read_parquet('./classic_hdbscan_50.parquet')
    df_hdbscan_2 = pd.read_parquet('./hdbscan_100.parquet')
    df_kmeans = pd.read_parquet('./kmeans_100.parquet')

    # print(df_hdbscan_1.head())
    # print(df_hdbscan_2.head())
    # print(df_kmeans.head())

    return {
        "HDBSCAN Classic Features": df_hdbscan_1,
        "KMeans Dino": df_kmeans,
        "HDBSCAN Dino": df_hdbscan_2
    }

def plot_tsne_highlighted(df, name, highlighted_clusters, cluster_names=None):
    """Plots the TSNE highlighting specific clusters."""
    plt.figure(figsize=(12, 8))
    
    # Gray background for all points
    base = df.copy()
    base_color = 'lightgray'
    plt.scatter(base['x'], base['y'], color=base_color, s=10, alpha=0.3)

    palette = sns.color_palette("tab10", len(highlighted_clusters))

    for i, cluster_id in enumerate(highlighted_clusters):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Plotar pontos do cluster
        plt.scatter(cluster_df['x'], cluster_df['y'],
                    color=palette[i],
                    s=25, alpha=0.9)
        
        # Adicionar nome do cluster no centro do grupo
        if len(cluster_df) > 0:
            # Usar mediana ao invés de média para ser mais robusto contra outliers
            center_x = cluster_df['x'].median()
            center_y = cluster_df['y'].median()
            
            # Usar nome do cluster se fornecido
            if cluster_names and cluster_id in cluster_names:
                label_text = f'{cluster_id}\n{cluster_names[cluster_id]}'
            else:
                label_text = f'{cluster_id}'
            
            plt.text(center_x, center_y, label_text, 
                    fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor='black',
                             alpha=0.4))

    plt.title(f"{name} - Highlight for clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_")}_highlighted_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {name.replace(' ', '_')}_highlighted_clusters.png")


def get_user_parameters():
    """Gets from the user the number of clusters and images per cluster."""
    try:
        num_clusters = int(input("How many clusters to show per grouping? "))
        num_images = int(input("How many images per cluster to show? "))
    except ValueError:
        print("Invalid input. Using default values: 3 clusters, 10 images.")
        num_clusters = 3
        num_images = 10
    return num_clusters, num_images

def plot_tsne(df, name):
    """Plots the TSNE graph with colors by cluster."""
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab20", df['cluster'].nunique())

    def color_map(c):
        return 'lightgray' if c == -1 else palette[c % len(palette)]

    colors = df['cluster'].apply(color_map)
    plt.scatter(df['x'], df['y'], c=colors, s=10, alpha=0.6, edgecolor='none')
    plt.title(f"{name} - TSNE visualization of clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_")}_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {name.replace(' ', '_')}_tsne.png")

def analyze_clusters(dataframes, num_clusters, num_images):
    """Analisa os clusters e exibe thumbnails e IDs."""
    # Mapeamento de números para nomes dos clusters do HDBSCAN Dino
    cluster_names = {
        25: "Desenhos/Conteúdo Infantil",
        3: "Xadrez", 
        16: "Violão/Guitarra/Música",
        15: "Fone de Ouvido/React/Música",
        12: "Armas",
        11: "Cabelo/Trump",
        7: "Unhas/Cosméticos",
        19: "Árabe",
        18: "Celular/Notebook/Apple",
        32: "Fogueira/Lareira",
        33: "Chuva/Natureza",
        30: "Avião",
        4: "Academia",
        27: "Navios",
        29: "Comidas",
        5: "Luta/Boxe",
        14: "Dirigindo",
        35: "Fórmula 1",
        17: "Planeta",
        24: "???"
    }
    
    # Processar apenas HDBSCAN Dino
    name = "HDBSCAN Dino"
    df = dataframes[name]
    
    print(f"\n{name} - Total points: {len(df)}")
    
    valid_clusters = df[df['cluster'] != -1]
    cluster_counts = valid_clusters['cluster'].value_counts().head(num_clusters)

    highlighted_clusters = []

    for cluster_id, count in cluster_counts.items():
        # Usar nome do cluster se tiver mapeamento
        if cluster_id in cluster_names:
            cluster_display = f"{cluster_id} - {cluster_names[cluster_id]}"
        else:
            cluster_display = str(cluster_id)
            
        print(f"\n{name} - Cluster {cluster_display} - {count} vídeos")
        ids = valid_clusters[valid_clusters['cluster'] == cluster_id]['id'].tolist()
        print("Video IDs:")
        for vid in ids[:num_images]:
            print(f"https://www.youtube.com/watch?v={vid}")
        highlighted_clusters.append(cluster_id)

    # Gerar apenas o gráfico t-SNE destacado
    plot_tsne_highlighted(df, name, highlighted_clusters, cluster_names)


def display_thumbnails(df, cluster_id, max_imgs=10, cluster_name=None):
    """Displays up to max_imgs thumbnails from the specified cluster."""
    subset = df[df['cluster'] == cluster_id].head(max_imgs)
    total = len(subset)
    
    n_cols = 5
    n_rows = (total + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for i, (_, row) in enumerate(subset.iterrows()):
        plt.subplot(n_rows, n_cols, i + 1)
        image_path = os.path.join('.', row['image_path'].lstrip('/'))
        try:
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(row['id'], fontsize=8)
        except Exception as e:
            plt.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center')
            plt.axis('off')
    
    # Usar nome do cluster se fornecido
    if cluster_name:
        title = f'Cluster {cluster_id} - {cluster_name} - Displaying {total} thumbnails'
    else:
        title = f'Cluster {cluster_id} - Displaying {total} thumbnails'
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster_id}_{cluster_name.replace("/", "_").replace("?", "")}_thumbnails.png' if cluster_name else f'cluster_{cluster_id}_thumbnails.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: cluster_{cluster_id}_{cluster_name.replace('/', '_').replace('?', '') if cluster_name else ''}_thumbnails.png")

def main():
    dataframes = load_dataframes()
    num_clusters, num_images = get_user_parameters()
    analyze_clusters(dataframes, num_clusters, num_images)

if __name__ == "__main__":
    main()
