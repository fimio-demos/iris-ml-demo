from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_decision_tree():
    """
    Trains a decision tree classifier on the Iris dataset and makes predictions on the test set.

    Returns:
        y_test (array-like): The true target values for the test set.
        predictions (array-like): The predicted target values for the test set.
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    # Create a decision tree classifier and fit it to the training data
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    # Make predictions on the test set
    predictions = classifier.predict(x_test)

    return y_test, predictions


if __name__ == "__main__":
    y_test, predictions = train_decision_tree()
    acc = accuracy_score(y_test,predictions)
    print(y_test, predictions, acc)