from flask import Flask, jsonify, Response, render_template
import threading
import numpy as np
import tensorflow as tf
import cv2
import time

# Configuração do Flask
app = Flask(__name__)

# Carregar o modelo TFLite uma vez
interpreter = tf.lite.Interpreter(model_path="model1.tflite")
interpreter.allocate_tensors()

# Obter detalhes da entrada e saída do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inicializar a câmera com resolução reduzida
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Obter o número de classes com base no tamanho da saída do modelo
interpreter.invoke()  # Executa uma inferência inicial para obter a forma de saída
output_data = interpreter.get_tensor(output_details[0]['index'])
num_classes = output_data.shape[1]

# Gerar nomes de classes genéricos
class_names = [f"Classe {i}" for i in range(num_classes)]

# Variável global para armazenar o último resultado
latest_result = {"main_class": None, "main_probability": 0, "other_classes": []}

def continuous_image_processing():
    global latest_result
    # Controle a frequência de processamento para economizar CPU e memória
    frame_skip_interval = 6  # Processar a cada 5 frames
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem")
            time.sleep(1)
            continue

        # Pular frames para reduzir carga (processa a cada 5 frames)
        frame_counter += 1
        if frame_counter % frame_skip_interval != 0:
            continue

        # Redimensionar o frame para o tamanho esperado pelo modelo
        input_shape = input_details[0]['shape']
        resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
        
        # Normalizar o frame para o modelo
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

        # Executar a inferência
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Obter os resultados
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = tf.nn.softmax(output_data[0]).numpy()

        # Identificar a classe com maior probabilidade
        result = np.argmax(probabilities)
        
        # Armazenar o último resultado
        latest_result = {
            "main_class": class_names[result],
            "main_probability": float(probabilities[result]),
            "other_classes": [
                {"class": class_names[i], "probability": float(prob)}
                for i, prob in enumerate(probabilities) if i != result
            ]
        }
        time.sleep(0.3)  # Pequena pausa para aliviar a carga de CPU

# Iniciar o processamento de imagem contínuo em uma thread separada
thread = threading.Thread(target=continuous_image_processing, daemon=True)
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Retorna o último resultado processado
    return jsonify(latest_result)

# Função geradora para o feed de vídeo
def generate_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Codificar o frame em JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Usar o formato de streaming MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Rota para o feed de vídeo
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
