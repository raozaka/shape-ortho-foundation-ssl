from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def hospital_leakage_test(embeddings, hospital_labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, hospital_labels)

    preds = clf.predict(embeddings)
    return accuracy_score(hospital_labels, preds)
