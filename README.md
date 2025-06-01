# emotional_speech_synthesis
Данный репозиторий содержит код для инференса и обучения моделей синтеза эмоциональной речи, представленных в ВКР. 

## Tacotron 2
Исходный код представляет собой комбинацию репозиториев https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2 и https://github.com/KinglittleQ/GST-Tacotron с доработанной функцией инференса и добавленным модулем на классификацию эмоций.  
Установка:
1. Создать окружение с python 3.10
2. ```cd Tacotron2```
3. ```pip install -r requirements.txt```  
4. Скачать веса моделей Tacotron и WaveGlow: https://drive.google.com/drive/folders/1SEGdztC2P9-h8MtAX8ZQ0a6Shk0J8hEy?usp=sharing

Инференс.  
1. Указать путь до весов такотрона и вокодера с помощью флагов --tacotron2 и --waveglow соответственно
2. Указать путь до файла с текстом, который нужно синтезировать, с помощью флага -i 
3. Указать путь до директории, куда сохранится выход модели, с помощью флага -o
4. Указать флаг --cpu, если на компьютере нет видеокарты  
  
Пример команды:
```
python inference.py --tacotron2 weights/checkpoint_Tacotron2_7255.pt --waveglow weights/waveglow_1076430_14000_amp --wn-channels 256 -i phrases/phrase.txt -o output --cpu
```

## Orpheus  
Для обучения модели Orpheus использовался оригинальный репозиторий без модификаций https://github.com/canopyai/Orpheus-TTS , поэтому в соответствующей директории предоставлен только ноутбук с кодом для инференса. Модель основана на библиотеке transformers, для инференса код модели не требуется.
Установка:
1. Создать окружение с python 3.12
2. ```cd Orpheus```
3. ```pip install -r requirements.txt```  

Инференс:  
см. inference.ipynb