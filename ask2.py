import numpy as np
import matplotlib.pyplot as plt

# Σταθερά για ανοχή (e = 10^-3)
e = 1e-3

#Αρχικοποίηση του SSE
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
    global sse_history #χρηση της global μεταβλητης
    sse_history = []  # Αποθήκευση SSE σε κάθε επανάληψη

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


        SSE = 0
        for k in range(K):
            points_in_cluster = Data[IDC == k]
            center = new_ClusterCenters[k]
            SSE += np.sum((np.linalg.norm(points_in_cluster - center, axis=1))**2)
        sse_history.append(SSE)

        # Έλεγχος σύγκλισης με βάση την σταθερά e
        if np.all(np.linalg.norm(new_ClusterCenters - ClusterCenters, axis=1) < e):
            break
        """
        Η np.linalg.norm(x, ord=None, axis=None, keepdims=False) είναι μια συνάρτηση της NumPy που υπολογίζει τη νόρμα 
        (δηλαδή, το μέγεθος ή την απόσταση) ενός πίνακα ή διάνυσματος.
        Στην συγκεκριμενη περιπτωση:
        x: Ο πίνακας ή το διάνυσμα για τον οποίο υπολογίζεται η νόρμα δηλαδή η διαφορά παλιών και καινούριων κέντρων
        το axis=1 για να γινει ο υπολογισμος κατα μηκος της δευτερης γραμμης 
        np.all(condition, axis=None) οπου το condition ειναι το παραπανω και επιστρέφει True/False
        """

        ClusterCenters = new_ClusterCenters

    return ClusterCenters, IDC

# Συνάρτηση για δημιουργία δεδομένων
def generate_data():
    """
    np.random.multivariate_normal(mean, cov, size):
    mean: Το διάνυσμα με τους μέσους όρους (μ) για κάθε διάσταση.
    cov: Ο πίνακας διασποράς-συνδιασποράς (Σ).
    size: Ο αριθμός σημείων που θέλουμε να δημιουργήσουμε.
    """
    np.random.seed(0)
    x1 = np.random.multivariate_normal([4, 0], [[0.29, 0.4], [0.4, 4]], 50)
    x2 = np.random.multivariate_normal([5, 7], [[0.29, 0.4], [0.4, 0.9]], 50)
    x3 = np.random.multivariate_normal([7, 4], [[0.64, 0], [0, 0.64]], 50)
    return np.vstack([x1, x2, x3]) #Συνδυάζει τις κατανομές σε έναν πίνακα που έχει διαστάσεις150 επι 2 (150 σημεία, 2 χαρακτηριστικά).

# Συνάρτηση για γραφική παράσταση
def plot_results(Data, ClusterCenters, IDC, K):
    """
    Οπτικοποιεί τα αποτελέσματα του k-means clustering.
    """
    colors = ['red', 'green', 'blue']
    for k in range(K):
        plt.scatter(Data[IDC == k][:, 0], Data[IDC == k][:, 1], c=colors[k], label=f'Cluster {k+1}')
        """
        Για τα δεδομένα οπου:
        Data[IDC == k][:, 0]: Οι x συντεταγμένες των σημείων της ομάδας k.
        Data[IDC == k][:, 1]: Οι y συντεταγμένες των σημείων της ομάδας k.
        c=colors[k]: Καθορίζει το χρώμα για την ομάδα k.
        label=f'Cluster {k+1}': Προσθέτει μια ετικέτα (label) για την ομάδα k.
        """
    plt.scatter(ClusterCenters[:, 0], ClusterCenters[:, 1], c='black', marker='+', s=200, label='Centers')
    """
    Για τα κέντρα όπου:
    ClusterCenters[:, 0]: Οι xx-συντεταγμένες των κέντρων.
    ClusterCenters[:, 1]: Οι yy-συντεταγμένες των κέντρων.
    c='black': Τα κέντρα εμφανίζονται με μαύρο χρώμα.
    marker='+': Το σύμβολο που χρησιμοποιείται για τα κέντρα.
    TODO:θελει διαφορετικα συμβολα για καθε ομαδα
    s=200: Το μέγεθος των συμβόλων.
    """
    plt.legend()
    plt.title('K-means Clustering')
    plt.xlabel('Χ')
    plt.ylabel('Υ')
    plt.show()

    # Συνάρτηση για γράφημα SSE
def plot_sse(sse_history):
    """
    Δημιουργεί γράφημα του Sum of Squared Error (SSE) σε κάθε επανάληψη.
    """
    plt.plot(range(len(sse_history)), sse_history, marker='o')
    plt.title('Sum of Squared Error (SSE) vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.grid()
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


     # Βήμα 5: Γράφημα SSE
    print("Plotting SSE...")
    plot_sse(sse_history)

# Σημείο εκκίνησης του προγράμματος
if __name__ == "__main__":
    main()
