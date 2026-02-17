from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_evaluate(train_emb, train_labels, test_emb, test_labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_emb, train_labels)

    preds = knn.predict(test_emb)
    return accuracy_score(test_labels, preds)
