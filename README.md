# 🚦 Sistema Inteligente de Contagem de Veículos em um Semáforo

Este projeto tem como objetivo desenvolver um sistema automatizado capaz de **detectar, rastrear e contar veículos** que passam por um ponto específico em uma via urbana, levando em consideração o **status do semáforo** para evitar contagens incorretas.

---

## 📌 Funcionalidades

- ✅ Detecção de veículos em vídeo utilizando o modelo **YOLOv8**
- ✅ Rastreamento de veículos com **ByteTrack**
- ✅ Contagem de veículos **baseada em linha virtual**
- ✅ **Ignora veículos parados** no semáforo vermelho
- ✅ Geração de **relatórios em CSV** com:
  - ID dos veículos contados
  - Tempo em que cruzaram a linha
  - Status do semáforo no momento
  - Distribuição temporal dos veículos
  - Influência do semáforo no fluxo

---

## 🧠 Tecnologias Utilizadas

- [`Ultralytics YOLOv8`](https://docs.ultralytics.com)
- [`OpenCV`](https://opencv.org/)
- [`NumPy`](https://numpy.org/)
- [`ByteTrack`](https://github.com/ifzhang/ByteTrack)
- Python 3.8+

---

## 🧠 Explicação Técnica

### 📌 Por que usar YOLOv8?

O YOLO (You Only Look Once) é um dos algoritmos mais rápidos e precisos para detecção de objetos em tempo real. No projeto, ele é utilizado para identificar veículos (carros, motos, caminhões, ônibus) e semáforos em cada frame do vídeo.

-✅ Detecta múltiplos objetos com alta performance
-✅ Reconhece diferentes classes relevantes no contexto de tráfego
-✅ Fácil integração com OpenCV e Ultralytics

#### 🔁 Por que usar ByteTrack?
O ByteTrack é um algoritmo de rastreamento de múltiplos objetos (MOT) que associa detecções entre frames. Ele fornece um ID único para cada objeto, permitindo que o sistema saiba qual veículo já foi contado.

-🔍 Permite seguir cada veículo ao longo do tempo
-🚫 Evita contagens duplicadas
-📈 Garante rastreamento contínuo mesmo com oclusões parciais

### 🧮 Como a contagem de veículos é realizada?

A cada frame, os veículos são detectados e rastreados.
Calcula-se o centroide da bounding box de cada veículo.
O sistema define uma linha virtual de contagem (horizontal).
Se o centroide de um veículo cruza essa linha de cima para baixo:

-✅ E o semáforo está verde (ou "unknown")
-✅ E o ID ainda não foi contado
-👉 O veículo é contado

O tempo e status do semáforo no momento da contagem são armazenados para análise posterior.



## 🛠️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/sistema-contagem-veiculos.git
cd sistema-contagem-veiculos

pip install -r requirements.txt
python main.py


