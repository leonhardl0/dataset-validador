import os
import glob
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def carregar_e_dividir_dataset(pasta_base, tamanho_img=(64, 64), proporcao_teste=0.2, semente_aleatoria=42):
    imagens = []
    labels = []

    classes = sorted([d for d in os.listdir(pasta_base) if os.path.isdir(os.path.join(pasta_base, d))])
    print(f"Classes encontradas: {classes}")

    for nome_classe in classes:
        caminho_classe = os.path.join(pasta_base, nome_classe)
        caminhos_imagens = []
        for ext in ('*.png', '*.jpeg', '*.jpg'):
            caminhos_imagens.extend(glob.glob(os.path.join(caminho_classe, ext)))

        print(f"Carregando imagens de: {nome_classe} ({len(caminhos_imagens)} imagens)")

        for caminho_img in caminhos_imagens:
            try:
                img = Image.open(caminho_img).convert('L')
                img = img.resize(tamanho_img)
                imagens.append(np.array(img))
                labels.append(nome_classe)
            except Exception as e:
                print(f"Erro ao carregar imagem {caminho_img}: {e}")

    X = np.array(imagens)
    y = np.array(labels)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=semente_aleatoria, stratify=y
    )
    return X_treino, X_teste, y_treino, y_teste

def pre_processar_dados(X_treino, X_teste, y_treino, y_teste, dimensoes_img):
    X_treino_proc = X_treino.astype('float32') / 255.0
    X_teste_proc = X_teste.astype('float32') / 255.0

    X_treino_proc = np.expand_dims(X_treino_proc, axis=-1)
    X_teste_proc = np.expand_dims(X_teste_proc, axis=-1)

    encoder_labels = LabelEncoder()
    y_treino_codificado = encoder_labels.fit_transform(y_treino)
    y_teste_codificado = encoder_labels.transform(y_teste)

    num_classes = len(encoder_labels.classes_)
    y_treino_one_hot = to_categorical(y_treino_codificado, num_classes=num_classes)
    y_teste_one_hot = to_categorical(y_teste_codificado, num_classes=num_classes)

    print(f"Classes detectadas e mapeadas: {list(encoder_labels.classes_)}")
    input_shape = (dimensoes_img[1], dimensoes_img[0], 1)
    return X_treino_proc, X_teste_proc, y_treino_one_hot, y_teste_one_hot, num_classes, input_shape

def construir_modelo_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def treinar_modelo(modelo, X_treino, y_treino, X_teste, y_teste, epocas=50, tamanho_lote=32):
    historico = modelo.fit(
        X_treino, y_treino,
        epochs=epocas,
        batch_size=tamanho_lote,
        validation_data=(X_teste, y_teste),
        verbose=1
    )
    return historico

def avaliar(modelo, X_teste, y_teste):
    print("\nAvaliando o modelo no conjunto de teste...")
    perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
    print(f"Perda final no conjunto de teste: {perda:.4f}")
    print(f"Acurácia final no conjunto de teste: {acuracia:.4f}")

def main():
    pasta_base_dataset = "caminho/do/seu/dataset" # <-- mude aqui
    # exemplo: "C:/Users/seu_nome/Downloads/dataset-validador-main"
    dimensoes_imagens = (64, 64)
    epocas_treino = 50

    print("\n--- ETAPA 1: Carregando Dataset (Raio-X vs. Outros) ---")
    X_treino, X_teste, y_treino, y_teste = carregar_e_dividir_dataset(pasta_base_dataset, dimensoes_imagens)

    if len(X_treino) == 0:
        print("Dataset não encontrado ou vazio. Verifique o caminho em 'pasta_base_dataset'. Encerrando.")
        return

    print("\n--- ETAPA 2: Pré-processando os Dados ---")
    X_treino_proc, X_teste_proc, y_treino_one_hot, y_teste_one_hot, num_classes, input_shape = pre_processar_dados(
        X_treino, X_teste, y_treino, y_teste, dimensoes_imagens
    )

    print("\n--- ETAPA 3: Construindo o Modelo Validador ---")
    modelo_validador = construir_modelo_cnn(input_shape, num_classes)

    print("\n--- ETAPA 4: Treinando o Modelo ---")
    treinar_modelo(modelo_validador, X_treino_proc, y_treino_one_hot, X_teste_proc, y_teste_one_hot, epocas=epocas_treino)

    print("\n--- ETAPA 5: Avaliando e Salvando o Modelo ---")
    avaliar(modelo_validador, X_teste_proc, y_teste_one_hot)

    modelo_validador.save("validador_raiox.keras")
    print("Modelo salvo como 'validador_raiox.keras'")

if __name__ == "__main__":
    main()