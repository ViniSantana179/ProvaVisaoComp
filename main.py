import cv2
import os
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models


def carregar_imagens(pasta, extensoes_permitidas={'.jpeg'}, tamanho_alvo=(128, 128)):
    imagens = {}
    if not os.path.isdir(pasta):
        print(f"Pasta não encontrada: {pasta}")
        return imagens

    for nome_arquivo in os.listdir(pasta):
        caminho_arquivo = os.path.join(pasta, nome_arquivo)
        _, extensao = os.path.splitext(nome_arquivo)
        
        if extensao.lower() in extensoes_permitidas:
            imagem = cv2.imread(caminho_arquivo)
            if imagem is not None:
                imagem_processada = pre_processar_imagem(imagem, tamanho_alvo)
                imagens[nome_arquivo] = imagem_processada
            else:
                print(f"Não foi possível carregar a imagem: {nome_arquivo}")
    return imagens

def pre_processar_imagem(imagem, tamanho=(128, 128)):
    if imagem is None:
        raise ValueError("A imagem não pode ser None.")
    
    imagem_redimensionada = cv2.resize(imagem, tamanho)
    imagem_gaussiana = cv2.GaussianBlur(imagem_redimensionada, (5, 5), 0) 
    
    if len(imagem_gaussiana.shape) == 3 and imagem_gaussiana.shape[2] == 3:
        imagem_gray = cv2.cvtColor(imagem_gaussiana, cv2.COLOR_BGR2GRAY)
        imagem_equalizada = cv2.equalizeHist(imagem_gray)
        imagem_final = cv2.cvtColor(imagem_equalizada, cv2.COLOR_GRAY2BGR) 
    else:
        imagem_final = cv2.equalizeHist(imagem_gaussiana)
        
    return imagem_final

def ajustar_tamanho_cnn(imagem, tamanho=(32, 32)):
    if imagem is None:
        raise ValueError("A imagem não pode ser None.")
    return cv2.resize(imagem, tamanho)

def preparar_imagem_cnn(imagem):
    img_ajustada = ajustar_tamanho_cnn(imagem)
    
    if len(img_ajustada.shape) == 2:
        img_rgb = cv2.cvtColor(img_ajustada, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB)
        
    img_normalizada = img_rgb.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalizada, axis=0)
    return img_batch

def carregar_dados():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    classes_interesse = [3, 5]

    train_mask = np.isin(y_train, classes_interesse)
    test_mask = np.isin(y_test, classes_interesse)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    y_train = np.where(y_train == 3, 0, 1)
    y_test = np.where(y_test == 3, 0, 1)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)

def construir_modelo(input_shape=(32, 32, 3), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def treinar_modelo(model, x_train, y_train, epochs=10, batch_size=64):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, validation_split=0.1)
    return history

def avaliar_modelo(model, x_test, y_test, class_names):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))


if __name__ == "__main__":
    class_names = ['gato', 'cachorro']

    (x_train, y_train), (x_test, y_test) = carregar_dados()

    modelo = construir_modelo()
    historia = treinar_modelo(modelo, x_train, y_train, epochs=10)

    avaliar_modelo(modelo, x_test, y_test, class_names)

    imagens = carregar_imagens('imagens', {'.jpeg'})

    for nome, img in imagens.items():
        try:
            img_preparada = preparar_imagem_cnn(img) 
            pred = modelo.predict(img_preparada)
            classe_idx = np.argmax(pred)
            prob = np.max(pred)
            print(f"{nome} -> Classe: {class_names[classe_idx]} (confiança {prob:.2f})")
        except Exception as e:
            print(f"Erro ao processar {nome}: {e}")