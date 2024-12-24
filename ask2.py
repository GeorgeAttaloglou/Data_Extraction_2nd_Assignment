import numpy as np
import matplotlib.pyplot as plt

# Σταθερά για ανοχή (e = 10^-3)
e = 1e-3

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
    np.random.seed(42)
    """
    Για αναπαραγωγιμότητα διότι χωρίς seed οι τιμές είναι διαφορετικές κάθε φορά 
    και υπηρχε κινδυνος ατελείωτης επανάληψης (πιθανό addition μια max_itterations παράμετρος)
    """
    indices = np.random.choice(Data.shape[0], K, replace=False)  # Τυχαία αρχικά κέντρα
    """
    Data.shape[0] επιστρέφει τον αριθμό των γραμμών του Data (το σύνολο απο το οποίο θα γίνει επιλογή)
    K ειναι ο αριθμός των κέντρων που θέλουμε να επιλέξουμε
    replace=False σημαίνει ότι δεν επιτρέπεται η επιλογή του ίδιου αριθμού 
    """
    ClusterCenters = Data[indices]

    while True:
        # Υπολογισμός αποστάσεων και ανάθεση σημείων
        IDC = []
        for point in Data:
            distances = [distance(point, center) for center in ClusterCenters] #Βοηθτιτική λίστα η οποια κραταει τις αποστάσεις του σημείου από κάθε κέντρο
            IDC.append(np.argmin(distances))   #Ανάθεση σημείου στο κοντινότερο κέντρο
        IDC = np.array(IDC) #Μετατροπή σε numpy array (με σκοπο ευκολοτερης χρήσης μαθηματικών συναρτήσεων)

        # Υπολογισμός νέων κέντρων
        new_ClusterCenters = np.array([Data[IDC == k].mean(axis=0) for k in range(K)])
        """
        - Υπολογίζει τα νέα κέντρα (centroids) για κάθε ομάδα.
        - Επιλέγει όλα τα σημεία που ανήκουν σε μια ομάδα και υπολογίζει τον μέσο όρο των χαρακτηριστικών τους.
        - Επαναλαμβάνει για όλες τις ομάδες (clusters) και δημιουργεί έναν πίνακα με τα νέα κέντρα.
        """

        # Έλεγχος σύγκλισης με βάση την σταθερά e
        if np.all(np.linalg.norm(new_ClusterCenters - ClusterCenters, axis=1) < e):
            break

        ClusterCenters = new_ClusterCenters

    return ClusterCenters, IDC

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
def plot_results(Data, ClusterCenters, IDC, K):
    """
    Οπτικοποιεί τα αποτελέσματα του k-means clustering.
    """
    colors = ['red', 'green', 'blue']
    for k in range(K):
        plt.scatter(Data[IDC == k][:, 0], Data[IDC == k][:, 1], c=colors[k], label=f'Cluster {k+1}')
    plt.scatter(ClusterCenters[:, 0], ClusterCenters[:, 1], c='black', marker='+', s=200, label='Centers')
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
    Data = generate_data()

    # Βήμα 2: Ορισμός αριθμού ομάδων (K)
    K = 3

    # Βήμα 3: Εκτέλεση k-means
    print("Running k-means clustering...")
    ClusterCenters, IDC = mykmeans(Data, K)

    # Βήμα 4: Οπτικοποίηση αποτελεσμάτων
    print("Clustering completed. Plotting results...")
    plot_results(Data, ClusterCenters, IDC, K)

# Σημείο εκκίνησης του προγράμματος
if __name__ == "__main__":
    main()
