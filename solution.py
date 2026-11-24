import argparse
from pathlib import Path

import cv2


def main():
    script_folder = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Находит контуры изображения и рисует их зеленым цветом."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(script_folder / "input.jpg"),
        help="Путь к изображению (по умолчанию: input.jpg рядом со скриптом)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    original = cv2.imread(str(image_path))
    if original is None:
        raise SystemExit(
            f"Не удалось открыть изображение: {image_path}. "
            "Проверьте путь и формат файла."
        )

    # Переводим цветное изображение в оттенки серого.
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Размываем изображение, чтобы уменьшить мелкий шум.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Находим границы методом Canny и контуры на их основе.
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Рисуем все контуры зеленым цветом поверх исходного цветного изображения.
    cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

    output_path = script_folder / "result.jpg"
    if not cv2.imwrite(str(output_path), original):
        raise SystemExit(f"Не удалось сохранить файл: {output_path}")

    print(f"Найдено контуров: {len(contours)}")
    print(f"Результат сохранен: {output_path}")


if __name__ == "__main__":
    main()
