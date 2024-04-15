import telebot
from my_detector import Detector
import numpy as np
import cv2, requests
from datetime import datetime, timedelta
from config import TOKEN
from classify import process_license_plate
from requests.exceptions import ReadTimeout


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
    
    classes = []
    
    for text in predicted_image_texts:
        classes.append(process_license_plate(text))
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
            bot.send_message(message.chat.id, 'Номер: '+ predicted_image_texts[i])
            bot.send_message(message.chat.id, f'Номер является: {classes[i]}')
            print(images[i].shape, predicted_image_texts[i], datetime.now() + timedelta(hours=7))
    except Exception as e:
        print(f"Ошибка плашка: {e}")
        
@bot.message_handler(content_types=['video', 'animation'])
def handle_media(message):
    bot.reply_to(message, "Обрабатывается видео...")
    if message.content_type == 'video':
        media_id = message.video.file_id
    else:
        media_id = message.animation.file_id
        
    try:
        media_info = bot.get_file(media_id)
    except telebot.apihelper.ApiTelegramException as e:
        if "file is too big" in str(e):
            print("Файл слишком большой: ", e)
            bot.send_message(message.chat.id, "Файл слишком большой, попробуйте видео меньшего размера.")
            return
    media_content = bot.download_file(media_info.file_path)

    if media_content:
        with open("media.mp4", "wb") as media_file:
            media_file.write(media_content)
        detector.video_call("media.mp4")
        try:
            bot.send_video(message.chat.id, open('out.mp4', 'rb'))
        except telebot.apihelper.ApiTelegramException as e:
            print("Невозможно отправить видео, файл слишком большой", e)
            bot.send_message(message.chat.id, "Видео обработано, но оно слишком большое.\nНевозможно отправить. Попробуйте видео меньшего размера.")
            return
    else:
        bot.reply_to(message, "Не удалось загрузить медиа")
        
@bot.message_handler()
def handle_content(message):
    content_type = message.content_type
    print(content_type)
    bot.reply_to(message, f"Бот умеет работать с изображениями и видео. Вы прислали контент типа: {content_type}")
        
    # Запуск бота
# bot.polling()

if __name__ == "__main__":
    while True:
        try:
            bot.polling()
        except ReadTimeout as e:
            print("Ошибка чтения времени ожидания:", e)
            pass