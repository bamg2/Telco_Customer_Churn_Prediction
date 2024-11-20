import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(df, column, hue):
    """Plots a distribution with respect to a given hue."""
    sns.countplot(data=df, x=column, hue=hue)
    plt.title(f'{column} Distribution by {hue}')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plots the confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
