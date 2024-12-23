import numpy as np
import matplotlib.pyplot as plt

# Συνάρτηση απόστασης
def distance(point1, point2):
    """
    Υπολογίζει την Ευκλείδεια απόσταση μεταξύ δύο σημείων.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Συνάρτηση για k-means
def mykmeans(data, K, max_iter=100, tol=1e-3):
    """
    Υλοποιεί τον αλγόριθμο k-means clustering.
    """
    np.random.seed(42)  # Για αναπαραγωγιμότητα
    indices = np.random.choice(data.shape[0], K, replace=False)  # Τυχαία αρχικά κέντρα
    centers = data[indices]

    for _ in range(max_iter):
        # Υπολογισμός αποστάσεων και ανάθεση σημείων
        labels = []
        for point in data:
            distances = [distance(point, center) for center in centers]
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        # Υπολογισμός νέων κέντρων
        new_centers = np.array([data[labels == k].mean(axis=0) for k in range(K)])

        # Έλεγχος σύγκλισης
        if np.all(np.linalg.norm(new_centers - centers, axis=1) < tol):
            break
        centers = new_centers

    return centers, labels

# Συνάρτηση για δημιουργία δεδομένων
def generate_data():
    """
    Δημιουργεί τυχαία δεδομένα για δοκιμή του αλγορίθμου k-means.
    """
    np.random.seed(0)
    x1 = np.random.multivariate_normal([4, 0], [[0.29, 0.4], [0.4, 4]], 50)
    x2 = np.random.multivariate_normal([5, 7], [[0.29, 0.4], [0.4, 0.9]], 50)
    x3 = np.random.multivariate_normal([7, 4], [[0.64, 0], [0, 0.64]], 50)
    return np.vstack([x1, x2, x3])

# Συνάρτηση για γραφική παράσταση
def plot_results(data, centers, labels, K):
    """
    Οπτικοποιεί τα αποτελέσματα του k-means clustering.
    """
    colors = ['red', 'green', 'blue']
    for k in range(K):
        plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], c=colors[k], label=f'Cluster {k+1}')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='+', s=200, label='Centers')
    plt.legend()
    plt.title('K-means Clustering')
    plt.xlabel('Χ1')
    plt.ylabel('Χ2')
    plt.show()

# Συνάρτηση main
def main():
    """
    Εκτελεί τη ροή του προγράμματος.
    """
    # Βήμα 1: Δημιουργία δεδομένων
    data = generate_data()

    # Βήμα 2: Ορισμός αριθμού ομάδων (K)
    K = 3

    # Βήμα 3: Εκτέλεση k-means
    print("Running k-means clustering...")
    centers, labels = mykmeans(data, K)

    # Βήμα 4: Οπτικοποίηση αποτελεσμάτων
    print("Clustering completed. Plotting results...")
    plot_results(data, centers, labels, K)

# Σημείο εκκίνησης του προγράμματος
if __name__ == "__main__":
    main()
