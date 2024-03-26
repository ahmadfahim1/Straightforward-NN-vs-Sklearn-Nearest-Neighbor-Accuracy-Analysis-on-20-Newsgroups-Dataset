from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import timeit
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
#pprint(data_train.target_names)
#print(f'Total of {len(data_train.data)} posts in the dataset and the total size is {size_mb(data_train.data):.2f}MB')


#X_train = vectorizer.fit_transform(data_train.data)
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
y_train, y_test = data_train.target, data_test.target
#print(X_train.shape)
#print(y_train.shape)


# implementation of dummy classifier


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
start_time = timeit.default_timer()
y_pred_dummy = dummy_clf.predict(X_test)
dummy_time = timeit.default_timer() - start_time
accuracy_d = accuracy_score(y_test, y_pred_dummy)
print("Baseline classifier Accuracy:", accuracy_d)
print("Baseline classifier Computation Time:", dummy_time, "seconds")

# implementation of manual KNN 

def knn_classify(train_data, train_labels, test_data, k=1):
    predictions = []
    for i in range(len(test_data)):
        distances = np.sqrt(np.sum((train_data - test_data[i])**2, axis=1))
        nearest_neighbors = np.argsort(distances)[:k]
        neighbor_labels = train_labels[nearest_neighbors]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predictions.append(predicted_label)
    return predictions


sample_size = 100
random_indices = np.random.choice(len(data_train.data), size=sample_size, replace=False)
X_train_sample = X_train[random_indices]
y_train_sample = y_train[random_indices]


train_data = X_train_sample.toarray()
train_labels = y_train_sample
test_data = X_test.toarray()  # Use full test data
start_time_manual = timeit.default_timer()
predicted_labels = knn_classify(train_data, train_labels, test_data, k=1)
knn_manual_time = timeit.default_timer() - start_time_manual

correct_predictions = sum(1 for pred, true in zip(predicted_labels, y_test) if pred == true)
accuracy = correct_predictions / len(y_test)
print("Sample size for straightforward NN: ", sample_size)
print("Accuracy for straightforward NN:", accuracy)
print(f'Computation Time for straightforward NN:  {knn_manual_time} seconds')

#implementation of KNN using library function


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
start_time_knn = timeit.default_timer()
y_pred = knn.predict(X_test)
knn_time = timeit.default_timer() - start_time_knn
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of KNN using Library:", accuracy)
print(f'Computation Time of KNN using Library: {knn_time} seconds')
