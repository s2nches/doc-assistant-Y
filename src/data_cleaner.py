import pandas as pd
import re
import string
import os

# Определяем пути к файлам внутри папки data
input_file = "data/dataset.csv"
output_file = "data/dataset_clean.csv"

# Проверяем, существует ли исходный файл
if not os.path.exists(input_file):
    print(f"⚠️ Ошибка: Файл {input_file} не найден! Положите исходный dataset.csv в папку data.")
else:
    print("1. Загружаем исходный CSV...")
    df = pd.read_csv(input_file)


    def clean_text(text):
        if pd.isna(text):
            return ""

        text = str(text)

        # 1. Нижний регистр
        text = text.lower()

        # 2. Удаляем все знаки препинания
        punctuation = string.punctuation + '«»—–…'
        translator = str.maketrans('', '', punctuation)
        text = text.translate(translator)

        # 3. Оставляем только русские/английские буквы, цифры и пробелы
        text = re.sub(r'[^а-яёa-z\d\s]', '', text)

        # 4. Убираем лишние пробелы, табуляции и переносы строк
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    print("2. Очищаем тексты...")
    df['text'] = df['text'].apply(clean_text)

    print("3. Сохраняем...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("✅ Готово! Файл data/dataset_clean.csv создан (цифры сохранены).")
    print("\nПример обработанного текста:\n")
    print(df['text'].iloc[0][:500] + "...")