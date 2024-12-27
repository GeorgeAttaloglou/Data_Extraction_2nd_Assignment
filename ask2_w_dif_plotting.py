import numpy as np
import matplotlib.pyplot as plt

# Σταθερά για ανοχή (e = 10^-3)
e = 1e-3

# Global μεταβλητή για αποθήκευση του SSE
sse_history = []

# Συνάρτηση απόστασης
def distance(point1, point2):
    """
    Υπολογίζει την Ευκλείδεια απόσταση μεταξύ δύο σημείων.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Συνάρτηση για k-means
def mykmeans(Data, K):
    """
    Υλοποιεί τον αλγόριθμο k-means clustering.
    """
    global sse_history  # Χρήση της global μεταβλητής
    sse_history = []  # Επαναφορά της σε κενή λίστα

    np.random.seed(42)
    indices = np.random.choice(Data.shape[0], K, replace=False)  # Τυχαία αρχικά κέντρα
    ClusterCenters = Data[indices]

    iteration = 0  # Μετρητής επαναλήψεων
    while True:
        # Υπολογισμός αποστάσεων και ανάθεση σημείων
        IDC = []
        for point in Data:
            distances = [distance(point, center) for center in ClusterCenters]
            IDC.append(np.argmin(distances))
        IDC = np.array(IDC)

        # Υπολογισμός νέων κέντρων
        new_ClusterCenters = np.array([Data[IDC == k].mean(axis=0) for k in range(K)])

        # Υπολογισμός SSE
        SSE = 0
        for k in range(K):
            points_in_cluster = Data[IDC == k]
            center = new_ClusterCenters[k]
            SSE += np.sum((np.linalg.norm(points_in_cluster - center, axis=1))**2)
        sse_history.append(SSE)

        # Εκτύπωση SSE για την τρέχουσα επανάληψη
        print(f"Iteration {iteration + 1}: SSE = {SSE}")

        # Δημιουργία εναλλασσόμενων markers
        markers = ['o', '^', 's', 'p', 'D', '*', 'X', 'h']  # Διαφορετικά markers
        current_markers = [markers[(iteration + i) % len(markers)] for i in range(K)]  # Εναλλαγή markers

        # Οπτικοποίηση αποτελεσμάτων για κάθε επανάληψη
        plot_results(Data, new_ClusterCenters, IDC, K, current_markers)
        plt.pause(0.5)  # Μικρή παύση για να φαίνεται το γράφημα

        # Έλεγχος σύγκλισης
        if np.all(np.linalg.norm(new_ClusterCenters - ClusterCenters, axis=1) < e):
            break

        ClusterCenters = new_ClusterCenters
        iteration += 1  # Ενημέρωση μετρητή επαναλήψεων

    return ClusterCenters, IDC

# Συνάρτηση για δημιουργία δεδομένων
def generate_data():
    """
    Δημιουργεί τυχαία δεδομένα για clustering.
    """
    np.random.seed(0)
    x1 = np.random.multivariate_normal([4, 0], [[0.29, 0.4], [0.4, 4]], 50)
    x2 = np.random.multivariate_normal([5, 7], [[0.29, 0.4], [0.4, 0.9]], 50)
    x3 = np.random.multivariate_normal([7, 4], [[0.64, 0], [0, 0.64]], 50)
    return np.vstack([x1, x2, x3])

# Συνάρτηση για γραφική παράσταση
def plot_results(Data, ClusterCenters, IDC, K, markers):
    """
    Οπτικοποιεί τα αποτελέσματα του k-means clustering.
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    for k in range(K):
        # Σχεδίαση των σημείων κάθε cluster
        plt.scatter(Data[IDC == k][:, 0], Data[IDC == k][:, 1], c=colors[k % len(colors)], label=f'Cluster {k+1}')
        # Σχεδίαση του κέντρου του cluster
        plt.scatter(ClusterCenters[k, 0], ClusterCenters[k, 1], c='black', marker=markers[k], s=200, label=f'Center {k+1}')
    plt.legend()
    plt.title('K-means Clustering')
    plt.xlabel('Χ')
    plt.ylabel('Υ')
    plt.show(block=False)  # Συνεχής εμφάνιση γραφημάτων χωρίς διακοπή

# Συνάρτηση main
def main():
    """
    Εκτελεί τη ροή του προγράμματος.
    """
    # Βήμα 1: Δημιουργία δεδομένων
    Data = generate_data()

    # Βήμα 2: Ορισμός αριθμού ομάδων (K)
    K = 3

    # Βήμα 3: Εκτέλεση k-means
    print("Running k-means clustering...")
    ClusterCenters, IDC = mykmeans(Data, K)

    # Βήμα 4: Τελική Οπτικοποίηση
    print("Clustering completed. Plotting results...")
    plot_results(Data, ClusterCenters, IDC, K, ['+', 's', 'p'])

# Σημείο εκκίνησης του προγράμματος
if __name__ == "__main__":
    main()
