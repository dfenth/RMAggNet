import torch
import torchvision
import numpy as np

def transform_mnist_dataset(dataset):
    """
    Transform the MNIST like datasets, scaling the values and expanding dimensions
    
    Parameters:
    dataset (list of (numpy.ndarray, int)): The dataset to transform

    Returns:
    - ret_dataset (list of (numpy.ndarray, int)): The transformed dataset
    """
    
    ret_dataset = []
    for image, label in dataset:
        image = torch.tensor((np.expand_dims(image, axis=0)/255.), dtype=torch.float32)
        ret_dataset.append((image, label))
    
    return ret_dataset



def transform_cifar_dataset(dataset):
    """
    Transform the CIFAR like dataset with some preprocessing layers

    Parameters:
    - dataset (list of (numpy.ndarray, int)): The dataset to transform

    Returns:
    - prepared_dataset (list of (numpy.ndarray, int)): The transformed dataset
    """

    # Set up transform pipeline
    dataset_transforms = torch.nn.Sequential(
        #torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.pil_to_tensor(x).type(torch.float32)), 
        torchvision.transforms.Normalize([0.], [255.]),
        torchvision.transforms.RandomHorizontalFlip(p=0.3),
        torchvision.transforms.RandomRotation(10),
    )

    # Make it a JIT script!
    transform_script = torch.jit.script(dataset_transforms)

    prepared_dataset = []
    for image, label in dataset:
        image = torchvision.transforms.functional.pil_to_tensor(image).type(torch.float32)
        prepared_dataset.append((transform_script(image), label)) 
    
    return prepared_dataset