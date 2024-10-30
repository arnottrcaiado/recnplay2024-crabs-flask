# Projeto de Detecção com Flask e TensorFlow Lite

Este projeto utiliza Flask para fornecer uma API e uma interface web para um modelo de detecção de classes usando TensorFlow Lite. A aplicação realiza inferências em imagens capturadas pela câmera em tempo real, mostrando o feed de vídeo e os resultados da classificação em uma página HTML.

## Visão Geral do Projeto

1. **Inferência com TensorFlow Lite**: Carrega um modelo TFLite para realizar inferências sobre imagens capturadas pela câmera.
2. **Serviço Flask**: Fornece uma API para acessar as inferências e uma página web que exibe o feed de vídeo em tempo real e os resultados das classificações.
3. **Threading**: O processamento da câmera e a inferência do modelo ocorrem em uma thread separada para melhorar o desempenho.

## Configuração do Ambiente

### Requisitos

- **Python 3.8 ou superior**
- **pip** para instalação de pacotes

### Instalação em Diferentes Plataformas

#### macOS

1. **Instale o Homebrew** (caso ainda não tenha):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"



2. Instale o Python:

   brew install python
   
4. Clone o Repositório e Instale Dependências:

   git clone <URL_DO_REPOSITORIO>
   cd <PASTA_DO_PROJETO>
   python3 -m venv venv
   source venv/bin/activate
   pip install flask tensorflow opencv-python-headless numpy
   
Raspberry Pi
1. Atualize o Sistema e Instale Dependências:

   sudo apt update
   sudo apt upgrade
   sudo apt install python3-pip python3-venv

2. Clone o Repositório e Instale Dependências:

   git clone <URL_DO_REPOSITORIO>
   cd <PASTA_DO_PROJETO>
   python3 -m venv venv
   source venv/bin/activate
   pip install flask tensorflow opencv-python-headless numpy
   
3. Verifique o Acesso à Câmera:

   Certifique-se de que o Raspberry Pi tem acesso à câmera (ativa no raspi-config).

Windows
1. Instale o Python:

   Baixe e instale o Python em python.org.
   Certifique-se de selecionar a opção Add Python to PATH durante a instalação.

2. Clone o Repositório e Instale Dependências:

Abra o PowerShell ou o CMD:
bash
Copiar código
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_PROJETO>
python -m venv venv
venv\\Scripts\\activate
pip install flask tensorflow opencv-python-headless numpy
Configuração da Câmera:

Verifique se o índice da câmera está correto no código (cv2.VideoCapture(0)).
Estrutura do Projeto
A estrutura básica do projeto é a seguinte:

graphql
Copiar código
projeto/
├── app.py              # Código principal da aplicação Flask
├── model1.tflite       # Modelo TFLite para inferência (não incluído)
├── templates/
│   └── index.html      # Página HTML para exibir o feed de vídeo e resultados
└── README.md           # Documentação do projeto
Explicação do Código
Configuração do Flask e Variáveis Globais
python
Copiar código
from flask import Flask, jsonify, Response, render_template
import threading
import numpy as np
import tensorflow as tf
import cv2
import time

app = Flask(__name__)
O servidor Flask é configurado para fornecer uma API e servir a página HTML. Variáveis globais são importadas para capturar e processar dados de câmera e resultados de inferência.

Carregamento do Modelo TensorFlow Lite
python
Copiar código
interpreter = tf.lite.Interpreter(model_path="model1.tflite")
interpreter.allocate_tensors()
Carrega o modelo TFLite e prepara os tensores. O modelo precisa estar disponível no diretório do projeto com o nome model1.tflite.

Configuração da Câmera
python
Copiar código
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
Inicializa a câmera com resolução reduzida (320x240 pixels) para diminuir a carga de processamento. cv2.VideoCapture(0) captura da câmera principal (ajuste o índice se necessário).

Variáveis para Resultados e Processamento Contínuo
python
Copiar código
latest_result = {"main_class": None, "main_probability": 0, "other_classes": []}
class_names = [f"Classe {i}" for i in range(num_classes)]
latest_result armazena o último resultado de inferência e class_names gera nomes genéricos para as classes do modelo.

Função continuous_image_processing
python
Copiar código
def continuous_image_processing():
    global latest_result
    frame_skip_interval = 5
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem")
            time.sleep(1)
            continue

        frame_counter += 1
        if frame_counter % frame_skip_interval != 0:
            continue

        input_shape = input_details[0]['shape']
        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = tf.nn.softmax(output_data[0]).numpy()
        result = np.argmax(probabilities)
        
        latest_result = {
            "main_class": class_names[result],
            "main_probability": float(probabilities[result]),
            "other_classes": [
                {"class": class_names[i], "probability": float(prob)}
                for i, prob in enumerate(probabilities) if i != result
            ]
        }
        time.sleep(0.3)
Essa função processa imagens da câmera em uma thread separada, redimensionando o frame para o modelo, realizando inferência e atualizando latest_result com a classe e probabilidade.

Rotas do Flask
Rota /predict
python
Copiar código
@app.route('/predict', methods=['GET'])
def predict():
    return jsonify(latest_result)
A rota /predict retorna o último resultado processado como JSON:

main_class: Nome da classe principal.
main_probability: Probabilidade da classe principal.
other_classes: Lista das demais classes e probabilidades.
Rota /video_feed
python
Copiar código
def generate_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\\r\\n'
               b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
A rota /video_feed usa MJPEG para transmitir o feed de vídeo em tempo real, enviando os frames como imagens JPEG contínuas.

Interligando Dois Raspberry Pi
Conexão por Rede ou Cabo Ethernet
Para interligar dois Raspberry Pi, você pode utilizar:

Rede local (Wi-Fi): Ambos os dispositivos devem estar conectados à mesma rede Wi-Fi.
Cabo Ethernet: Conecte um cabo Ethernet entre os Raspberry Pi. Configure os endereços IP para que possam se comunicar, usando o IP do primeiro Raspberry Pi para acessar a API no segundo.
Configurando o Segundo Raspberry Pi para Acessar a API
No primeiro Raspberry Pi, execute o servidor Flask conforme as instruções acima.

No segundo Raspberry Pi, utilize o código abaixo para fazer uma solicitação à API /predict do primeiro Raspberry Pi:

python
Copiar código
import requests
import time

# IP do primeiro Raspberry Pi onde o servidor Flask está rodando
SERVER_IP = "http://<IP_DO_RASPBERRY_PI>:5001"

def get_prediction():
    try:
        response = requests.get(f"{SERVER_IP}/predict")
        if response.status_code == 200:
            prediction = response.json()
            print("Classe Principal:", prediction["main_class"])
            print("Probabilidade:", prediction["main_probability"])
            for other_class in prediction["other_classes"]:
                print(f"{other_class['class']}: {other_class['probability']}")
        else:
            print("Erro na resposta:", response.status_code)
    except Exception as e:
        print("Erro ao acessar o servidor:", e)

while True:
    get_prediction()
    time.sleep(2)  # Aguardar 2 segundos antes da próxima solicitação
Substitua <IP_DO_RASPBERRY_PI> pelo IP do Raspberry Pi que está rodando o servidor.
Esse código solicita a predição a cada 2 segundos e exibe os resultados.

# Notas para Produção
Para produção, use um servidor WSGI como Gunicorn para executar o Flask. Em um Raspberry Pi, 
recomenda-se usar um único worker:


gunicorn -w 1 -b 0.0.0.0:5001 app:app


# Considerações Finais
Este projeto fornece uma interface leve para inferência de modelos com TensorFlow Lite e Flask, utilizando uma câmera para capturar imagens em tempo real. O código é otimizado para ser executado em dispositivos com recursos limitados, como o Raspberry Pi, mas pode ser expandido conforme necessário para atender a requisitos específicos.
