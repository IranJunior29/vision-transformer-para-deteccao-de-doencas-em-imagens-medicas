# Imports
import os
import copy
import math
import torch
import timm
import torchvision
import pandas as pd
import numpy as np
from linformer import Linformer
from vit_pytorch.efficient import ViT
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Função para o plot da confusion matrix
def plot_cm(labels, predictions):

    conf_numpy = confusion_matrix(labels, predictions)
    conf_df = pd.DataFrame(conf_numpy, index = class_names, columns = class_names)
    plt.figure(figsize = (8,7))
    sns.heatmap(conf_df, annot = True, fmt = "d", cmap = "BuPu")
    plt.title('Confusion Matrix', fontsize = 15)
    plt.ylabel('Valor Real', fontsize = 14)
    plt.xlabel('Valor Previsto', fontsize = 14)


# Função para avaliar o modelo
def evaluate_model(model, dataloader, device):
    # Coloca o modelo em modo de avaliação
    modelo.eval()

    # Listas
    true_labels = []
    pred_labels = []

    # Faz as previsões a partir dos dados
    for inputs, labels in dataloader:
        # Envia imagens e labels para o device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        with torch.no_grad():
            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

    return true_labels, pred_labels

if __name__ == '__main__':

    processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = processing_device
    print(device)

    ''' Pré-Processamento das Imagens '''

    # Pasta de imagens (https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
    data_dir = "dados"

    # Dataset de imagens
    dataset_completo = datasets.ImageFolder(data_dir)

    # Nomes das classes
    class_names = dataset_completo.classes

    # Define o tamanho dos datasets
    full_size = len(dataset_completo)
    train_size = int(0.7 * full_size)
    val_size = full_size - train_size

    # Randomicamente divide as imagens nas amostras de dados
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_completo, [train_size, val_size])

    # Tamanhos dos datasets
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Processamento das imagens
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Média e desvio padrão para cada camada de cor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Aplica as transformações aos dados de treino
    train_dataset.dataset.transform = data_transforms['train']

    # Tamanho do batch
    batch_size = 32

    # Cria os dataloaders
    # Nota: se tiver erro aqui, remova o parâmetro num_workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)

    # Dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    ''' Modelagem - Carregando o Modelo ViT Pré-Treinado '''

    # Carregando modelo pré-treinado
    modelo = timm.create_model('vit_base_patch16_224', pretrained=True)

    # Número de atributos
    num_ftrs = modelo.head.in_features

    # Modifica a última camada de acordo com a tarefa de classificação
    modelo.head = nn.Linear(num_ftrs, len(class_names))

    # Envia o modelo para o device
    modelo = modelo.to(device)

    ''' Modelagem - Linformer Para Customização Eficiente do Modelo ViT '''

    # Atributos de entrada
    in_features = 128

    # Cria o objeto Linformer com os parâmetros de customização
    efficient_transformer = Linformer(dim=in_features, seq_len=49 + 1, depth=12, heads=8, k=64)

    # Define o modelo com os parâmetros de customização
    modelo = ViT(dim=in_features,
                 image_size=224,
                 patch_size=32,
                 num_classes=2,
                 transformer=efficient_transformer,
                 channels=3)

    # Número de atributos
    num_ftrs = in_features

    # Modificamos a última camada de acordo com a tarefa de classificação
    modelo.head = nn.Linear(num_ftrs, len(class_names))

    # Envia o modelo para o device
    modelo = modelo.to(device)

    ''' Treinamento do Modelo ViT Customizado '''

    # Função de erro
    criterion = nn.CrossEntropyLoss()

    # Otimizador
    optimizer = optim.Adam(modelo.parameters(), lr=3e-5)

    # Learning rate scheduler
    exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Parâmetros de controle
    num_epochs = 15
    best_model_wts = copy.deepcopy(modelo.state_dict())
    best_acc = 0.0

    # Listas para o histórico de treino
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Define qual fase estamos, treino ou validação
        for phase in ['train', 'val']:
            if phase == 'train':
                modelo.train()
            else:
                modelo.eval()

                # Contadores
            running_loss = 0.0
            running_corrects = 0

            # Iteração pelos dados
            for inputs, labels in dataloaders[phase]:

                # Envia imagens e labels para o device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zera os gradientes
                optimizer.zero_grad()

                # Forward propagation
                with torch.set_grad_enabled(phase == 'train'):

                    # Faz as previsões com o modelo
                    outputs = modelo(inputs)

                    # Obtém a maior probabilidade de classe
                    _, preds = torch.max(outputs, 1)

                    # Calcula o erro
                    loss = criterion(outputs, labels)

                    # Backpropagation e optimization somente em treino
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Calcula as estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calcula erro e acurácia
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            # Grava o histórico
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print('{} - Erro: {:.4f} Acurácia: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy do modelo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(modelo.state_dict())

        print()

    print('\nMelhor Acurácia em Validação: {:4f}'.format(best_acc))
    print('\nTreinamento Concluído')

    ''' Avaliação do Modelo '''

    # Época
    epoch = range(1, len(train_loss_history) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epoch, train_loss_history, label='Erro em Treino')
    ax[0].plot(epoch, val_loss_history, label='Erro em Validação')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Erro')
    ax[0].legend()

    ax[1].plot(epoch, train_acc_history, label='Acurácia em Treino')
    ax[1].plot(epoch, val_acc_history, label='Acurácia em Validação')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Acurácia')
    ax[1].legend()

    # Extrai valores reais e previsões de labels
    true_labels, pred_labels = evaluate_model(modelo, dataloaders['val'], device)

    # Calcula a confusion matrix
    cm_val = confusion_matrix(true_labels, pred_labels)
    a_val = cm_val[0, 0]
    b_val = cm_val[0, 1]
    c_val = cm_val[1, 0]
    d_val = cm_val[1, 1]

    # Calcula as métricas de performance

    # Accuracy
    acc_val = (a_val + d_val) / (a_val + b_val + c_val + d_val)

    # Error rate
    error_rate_val = 1 - acc_val

    # Sensitivity
    sen_val = d_val / (d_val + c_val)

    # Specificity
    sep_val = a_val / (a_val + b_val)

    # Precision
    precision_val = d_val / (b_val + d_val)

    # F1 score
    F1_val = (2 * precision_val * sen_val) / (precision_val + sen_val)

    # Coeficiente de Correlação de Matthews
    MCC_val = (d_val * a_val - b_val * c_val) / (
        np.sqrt((d_val + b_val) * (d_val + c_val) * (a_val + b_val) * (a_val + c_val)))

    # Print
    print("\n Sensitivity em validação:", sen_val,
          "\n Specificity em validação:", sep_val,
          "\n Accuracy em validação:", acc_val,
          "\n Error rate em validação:", error_rate_val,
          "\n Precision em validação:", precision_val,
          "\n F1 score em validação:", F1_val,
          "\n Matthews Correlation Coefficient (MCC) em validação:", MCC_val)

    # Confusion matrix
    plot_cm(true_labels, pred_labels)

    # Garantimos que o modelo esteja em modo de avaliação
    modelo.eval()

    # Dataloader de validação
    val_ds = dataloaders['val']

    # Listas para calcular AUC (Area Under The Curve)
    val_pre_auc = []
    val_label_auc = []

    # Loop para iterar através do conjunto de dados de validação (val_ds)
    for images, labels in val_ds:

        # Loop para iterar através de cada par de imagem e label no lote atual
        for image, label in zip(images, labels):
            # Adiciona uma dimensão extra, move a imagem para o dispositivo (geralmente uma GPU) e armazena em 'img_array'
            img_array = image.unsqueeze(0).to(device)

            # Utiliza o modelo para fazer uma previsão na imagem e armazena o resultado em 'prediction_auc'
            prediction_auc = modelo(img_array)

            # Adiciona a previsão à lista 'val_pre_auc', convertendo-a para uma matriz NumPy e obtendo
            # o segundo valor (índice 1)
            val_pre_auc.append(prediction_auc.detach().cpu().numpy()[:, 1])

            # Adiciona a etiqueta real ('label') à lista 'val_label_auc'
            val_label_auc.append(label.item())

    # Extrai o score
    auc_score_val = metrics.roc_auc_score(val_label_auc, val_pre_auc)

    print("O valor AUC para o conjunto de validação é:", auc_score_val)


    # Plot da ROC
    def plot_roc(name, labels, predictions, **kwargs):
        fp, tp, _ = metrics.roc_curve(labels, predictions)
        plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        ax = plt.gca()
        ax.set_aspect('equal')


    # Plot
    plot_roc('AUC em Validação: {0:.4f}'.format(auc_score_val), val_label_auc, val_pre_auc, color="red", linestyle='--')
    plt.legend(loc='lower right')

    # Salva o modelo
    torch.save(modelo, 'modelos/modelo_dsa.pt')

