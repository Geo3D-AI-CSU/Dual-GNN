from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix
import numpy as np



# Root Mean Square Error calculation
def calculate_rmse(predicted, target, mask):
    mask =mask.to('cpu')  # Transfer the mask to the CPU
    predicted = predicted[mask].detach().cpu().numpy()  # Use .detach() to detach the computational graph
    target = target[mask].detach().cpu().numpy()  # Separate target in the same manner
    return np.sqrt(mean_squared_error(target, predicted))


# Accuracy calculation
def calculate_accuracy(predicted, target, mask):
    # Ensure all tensors reside on the same device
    mask = mask.to('cpu')
    predicted = predicted[mask].detach().cpu()  # Remove gradient information and transfer to the CPU
    target = target[mask].detach().cpu()-1

    # Retrieve the predicted category (the index of the maximum value in each row)
    predicted_classes = predicted.argmax(dim=1)  # Select the category corresponding to the maximum probability at each node.
    correct = (predicted_classes == target).sum().item()  # Calculate the correct number of predictions
    total = target.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# R2 calculation
def calculate_r2(predicted, target, mask):
    # Select the required values from the mask and separate the computational graph.
    predicted = predicted[mask].detach().cpu().numpy()
    target = target[mask].detach().numpy()
    return r2_score(target, predicted)

# Confusion Matrix Calculation
def calculate_confusion_matrix(predicted, target, mask):
    # Filter out corresponding nodes using a mask.
    mask = mask.to('cpu')
    predicted = predicted[mask].detach().cpu().numpy()
    target = target[mask].detach().cpu().numpy()-1
    predicted_classes = predicted.argmax(axis=1)

    return confusion_matrix(target,  predicted_classes)
