import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

def validar_imagem_com_modelo(caminho_imagem, modelo, tamanho_img=(64, 64)):
    try:
        img = Image.open(caminho_imagem).convert('L')
        img = img.resize(tamanho_img)
        img_array = np.array(img).astype('float32')
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=[0, -1])

        predicoes = modelo.predict(img_array)
        indice_classe_predita = np.argmax(predicoes, axis=1)[0]

        classes_mapeadas = ['Outros', 'Raio-X']
        
        classe_predita = classes_mapeadas[indice_classe_predita]
        return classe_predita
        
    except Exception as e:
        print(f"Erro inesperado ao processar a imagem: {e}")
        return None

if __name__ == "__main__":
    caminho_sua_imagem = "caminho/da/sua/imagem" # <-- mude aqui
    # exemplo = "C:/Users/seu_nome/Downloads/imagem_fratura.png"
    caminho_modelo = "validador_raiox.keras"
    # nome do arquivo do modelo treinado (deve estar na mesma pasta)
    if not os.path.exists(caminho_modelo):
        print(f"\nERRO CRÍTICO: O modelo treinado '{caminho_modelo}' não foi encontrado.")
        print("Execute primeiro o script 'treinar_validador_raiox.py'.")
        exit()
    if not os.path.exists(caminho_sua_imagem):
         print(f"\nAVISO: O caminho da imagem '{caminho_sua_imagem}' é inválido.")
         exit()

    print(f"Carregando modelo validador: {caminho_modelo}...")
    modelo = load_model(caminho_modelo, compile=False)

    print(f"Analisando a imagem: {caminho_sua_imagem}...")
    classe = validar_imagem_com_modelo(caminho_sua_imagem, modelo)

    if classe:
        print("\n--------------------------------------")
        print("    Resultado da Validação")
        print("--------------------------------------")
        if classe == 'Raio-X':
            print("  Status: Imagem Válida")
            print("  Motivo: A imagem foi identificada como um Raio-X.")
        else:
            print("  Status: Imagem Inválida")
            print("  Motivo: A imagem não parece ser um Raio-X.")
        print("--------------------------------------\n")