import torch
from torchvision import transforms
from dataset import get_dataLoaders
from model import MultiLayerPerceptron, AlexNetExtension
from utils import set_seed, train_model, test_model, plot_results
import matplotlib.pyplot as plt
from guided_backprop_utils import *
from guided_backprop_utils import misc_functions
from guided_backprop_utils import guided_backprop
from guided_backprop_utils.guided_backprop import *
# import guided_backprop_utils.misc_functions
# from guided_backprop_utils.guided_backprop import *



# Hyperparameters ---------------
epochs = 20
lr = 1e-2
batch_size = 64
num_workers = 4
assignment_section = 2.5 # choose from [2.1, 2.2, 2.3, 2.4, 2.5]
# -------------------------------

if assignment_section < 2.3:
    model_architecture = None # choose from ['MLP', 'AlexNet-Backbone']
    if assignment_section == 2.1:
        model_architecture = 'MLP'
    elif assignment_section == 2.2:
        model_architecture = 'AlexNet-Backbone'

    # Set seed for reproducibility purposes
    set_seed(42)

    print("="*20, f"Using model_architecture: {model_architecture}", "="*20)
    if model_architecture == 'MLP':
        # Preprocessing for MLP
        preprocessing_transforms = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
        train_dataloader, val_dataloader, test_dataloader, unique_names, one_hot_celeb_name = get_dataLoaders(preprocessing_transforms, batch_size=batch_size, num_workers=num_workers)
        model = MultiLayerPerceptron(input_size=28*28*3, output_size=len(unique_names), hidden_size=300)
    elif model_architecture == 'AlexNet-Backbone':
        # Preprocessing for AlexNet
        preprocessing_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataloader, val_dataloader, test_dataloader, unique_names, one_hot_celeb_name = get_dataLoaders(preprocessing_transforms, batch_size=batch_size, num_workers=num_workers)
        model = AlexNetExtension(output_size=len(unique_names))
        
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # Train model
    train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist = train_model(model, train_dataloader, val_dataloader, epochs, optimizer, loss_func)

    # Test model
    test_loss, test_accuracy= test_model(model, test_dataloader, loss_func)
    print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")

    # Visualize Results
    plot_results(train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist)
elif assignment_section == 2.3:
    # Set seed for reproducibility purposes
    set_seed(42)

    def train_mlp(hidden_size, model_no):
        print("="*20, f"Training MLP model:{model_no} with hidden size: {hidden_size}", "="*20)
        # Preprocessing for MLP
        preprocessing_transforms = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])
        train_dataloader, val_dataloader, test_dataloader, unique_names, one_hot_celeb_name = get_dataLoaders(preprocessing_transforms, batch_size=batch_size, num_workers=num_workers)
        model = MultiLayerPerceptron(input_size=28*28*3, output_size=len(unique_names), hidden_size=300)
            
        optimizer = torch.optim.SGD(model.parameters(), lr)
        loss_func = torch.nn.CrossEntropyLoss()

        # Train MLP model
        train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist = train_model(model, train_dataloader, val_dataloader, epochs, optimizer, loss_func)

        # Test MLP model
        test_loss, test_accuracy= test_model(model, test_dataloader, loss_func)
        print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")
        return model
    
    # train model1
    model1 = train_mlp(300, 1)
    # train model2
    model2 = train_mlp(800, 2)

    # Obtain weighs of two neurons from each model
    model1_neuron1 = None
    model1_neuron2 = None
    for i, (name, param) in enumerate(model1.named_parameters()):
        if i == 0:
            model1_neuron1 = param[0]
            model1_neuron2 = param[1]
            break

    model2_neuron1 = None
    model2_neuron2 = None
    for i, (name, param) in enumerate(model2.named_parameters()):
        if i == 0:
            model2_neuron1 = param[0]
            model2_neuron2 = param[1]
            break

    model1_neuron1 = torch.permute(torch.reshape(model1_neuron1, (3, 28, 28)), (1, 2, 0)).cpu().detach()
    model1_neuron2 = torch.permute(torch.reshape(model1_neuron2, (3, 28, 28)), (1, 2, 0)).cpu().detach()
    model2_neuron1 = torch.permute(torch.reshape(model2_neuron1, (3, 28, 28)), (1, 2, 0)).cpu().detach()
    model2_neuron2 = torch.permute(torch.reshape(model2_neuron2, (3, 28, 28)), (1, 2, 0)).cpu().detach()
    # normalize weights
    model1_neuron1 -= model1_neuron1.min()
    model1_neuron1 /= model1_neuron1.max()
    # normalize weights
    model1_neuron2 -= model1_neuron2.min()
    model1_neuron2 /= model1_neuron2.max()
    # normalize weights
    model2_neuron1 -= model2_neuron1.min()
    model2_neuron1 /= model2_neuron1.max()
    # normalize weights
    model2_neuron2 -= model2_neuron2.min()
    model2_neuron2 /= model2_neuron2.max()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(model1_neuron1.numpy())
    plt.title("Model1 Neuron1 Weight Visualization")
    plt.subplot(2, 2, 2)
    plt.imshow(model1_neuron2.numpy())
    plt.title("Model1 Neuron 2 Weight Visualization")

    plt.subplot(2, 2, 3)
    plt.imshow(model2_neuron1.numpy())
    plt.title("Model2 Neuron 1 Weight Visualization")

    plt.subplot(2, 2, 4)
    plt.imshow(model2_neuron2.numpy())
    plt.title("Model2 Neuron 2 Weight Visualization")

    plt.show()


elif assignment_section == 2.4:
    # Set seed for reproducibility purposes
    set_seed(42)

    print("="*20, f"Using AlexNet with Different Classification Layer", "="*20)

    # Preprocessing for AlexNet
    preprocessing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataloader, val_dataloader, test_dataloader, unique_names, one_hot_celeb_name = get_dataLoaders(preprocessing_transforms, batch_size=batch_size, num_workers=num_workers)
    # Load pre-trained AlexNet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # Freeze pre-trained AlexNet Backbone
    for param in model.parameters():
        param.requires_grad = False
    
    new_classifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],
        torch.nn.Linear(4096, 40)
    )
    model.classifier = new_classifier
    for i, child in enumerate(model.classifier.children()):
        if i == len(list(model.classifier.children())) - 1:
            child.requires_grad = True # only unfreeze new Classification Layer
        else:
            child.requires_grad = False
    
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # Train model
    train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist = train_model(model, train_dataloader, val_dataloader, epochs, optimizer, loss_func)

    # Test model
    test_loss, test_accuracy= test_model(model, test_dataloader, loss_func)
    print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")

    # Visualize Results
    plot_results(train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist)

    # Show prediction results on a sample output
    model.eval()
    for data, label in test_dataloader:
        with torch.no_grad():
            preds = model(data) # [B, C]
            sample_pred = preds[0]
            sample_target = label[0]
            # Find name of the celebrity
            sample_predicted_name = None
            sample_target_name = None
            pred_ind = torch.argmax(sample_pred, dim=-1)
            sample_pred = torch.zeros_like(sample_pred, dtype=torch.uint8)
            sample_pred[pred_ind] = 1
            for k, v in one_hot_celeb_name.items():
                if torch.all(torch.eq(sample_pred, v)):
                    sample_predicted_name = k
                if torch.all(torch.eq(sample_target, v)):
                    sample_target_name = k
        

        # Visualize sample input
        sample_input = torch.permute(data[0], (1, 2, 0)).cpu().detach()
        # normalize weights
        sample_input -= sample_input.min()
        sample_input /= sample_input.max()

        plt.imshow(sample_input.numpy())
        plt.title(f"Prediction: {sample_predicted_name}, Target: {sample_target_name}")
        plt.show()
        break

elif assignment_section == 2.5:
    # Set seed for reproducibility purposes
    set_seed(42) 

    print("="*20, f"Using AlexNet with Different Classification Layer", "="*20)

    # Preprocessing for AlexNet
    preprocessing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataloader, val_dataloader, test_dataloader, unique_names, one_hot_celeb_name = get_dataLoaders(preprocessing_transforms, batch_size=batch_size, num_workers=num_workers)
    # Load pre-trained AlexNet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # Freeze pre-trained AlexNet Backbone
    for param in model.parameters():
        param.requires_grad = False
    
    new_classifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1],
        torch.nn.Linear(4096, 40)
    )
    model.classifier = new_classifier
    for i, child in enumerate(model.classifier.children()):
        if i == len(list(model.classifier.children())) - 1:
            child.requires_grad = True # only unfreeze new Classification Layer
        else:
            child.requires_grad = False
    
    optimizer = torch.optim.SGD(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # Train model
    train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist = train_model(model, train_dataloader, val_dataloader, epochs, optimizer, loss_func)

    # Test model
    test_loss, test_accuracy= test_model(model, test_dataloader, loss_func)
    print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")

    # Visualize Results
    plot_results(train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist)

    # Show prediction results on a sample output
    model.eval()
    sample_input_data = None
    sample_target = None
    for data, label in test_dataloader:
        with torch.no_grad():
            preds = model(data) # [B, C]
            sample_pred = preds[0]
            sample_target = label[0]
            # Find name of the celebrity
            sample_predicted_name = None
            sample_target_name = None
            pred_ind = torch.argmax(sample_pred, dim=-1)
            sample_pred = torch.zeros_like(sample_pred, dtype=torch.uint8)
            sample_pred[pred_ind] = 1
            for k, v in one_hot_celeb_name.items():
                if torch.all(torch.eq(sample_pred, v)):
                    sample_predicted_name = k
                if torch.all(torch.eq(sample_target, v)):
                    sample_target_name = k
        

        # Visualize sample input
        sample_input_data = data[0]
        sample_input = torch.permute(sample_input_data, (1, 2, 0)).cpu().detach()
        # normalize weights
        sample_input -= sample_input.min()
        sample_input /= sample_input.max()

        plt.imshow(sample_input.numpy())
        plt.title(f"Prediction: {sample_predicted_name}, Target: {sample_target_name}")
        plt.show()
        break

    # Guided Backpropagation:
    prep_img = torch.unsqueeze(sample_input_data, dim=0)
    prep_img.requires_grad = True
    file_name_to_export = "Guided Backprop results"
    pretrained_model = model
    target_class = torch.argmax(sample_target, dim=-1).cpu().detach().item()
    

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')


else:
    print(f" Invalid Section choice !")

    








