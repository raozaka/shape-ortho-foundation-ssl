import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings)

    plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='coolwarm')
    plt.title("t-SNE Embedding Visualization")
    plt.show()
