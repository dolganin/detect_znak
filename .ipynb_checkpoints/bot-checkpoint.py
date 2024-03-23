import telebot
from my_detector import Detector
import numpy as np
import cv2, requests
from datetime import datetime, timedelta

TOKEN = '7182931451:AAG8yDmbb86xknmOr_1ZmiRWsPYu203yt1U'

bot = telebot.TeleBot(TOKEN)

detector = Detector()

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Привет! Пришли мне фото с автомобилем.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    photo_path = file_info.file_path

    photo_url = f"https://api.telegram.org/file/bot{TOKEN}/{photo_path}"

    response = requests.get(photo_url, stream=True).raw
    img = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)

    image, image_zones, predicted_image_texts, confidences, _ = detector(image)
    
    
    try:
        image = np.array(image)
        _, encoded_image = cv2.imencode('.jpg', image)
        bot.send_photo(message.chat.id, encoded_image.tobytes())
    except Exception as e:
        print(f"Ошибка изображение: {e}")
        
    try:
        images = []
        
        for i in range(np.array(image_zones).shape[0]):
            image = image_zones[i]
            images.append(image)

        for i in range(len(images)):
            encoded_image = cv2.imencode('.jpg', images[i])[1].tobytes()
            bot.send_photo(message.chat.id, encoded_image)
            bot.send_message(message.chat.id, f'Вероятность определения ГРЗ: {round(confidences[0][i][0]*100, 2)}%')
            bot.send_message(message.chat.id, 'Номер: '+predicted_image_texts[i])
            print(images[i].shape, predicted_image_texts[i], datetime.now() + timedelta(hours=7))
    except Exception as e:
        print(f"Ошибка плашка: {e}")
        
    # Запуск бота
bot.polling()

