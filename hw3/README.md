# ДЗ 3 — Self-Supervised Learning

В этом задании продолжается работа с классификацией **CIFAR-10** и проводится эксперимент с self-supervised предобучением feature extractor, а затем — сравнение с обычным обучением на размеченных данных.

## Структура проекта

```text
hw3/
├── config.py                     # Общие настройки и пути
├── pretrain.py                   # Цикл SSL-предобучения (SimCLR)
├── main.py                       # CLI-скрипт запуска SSL-предобучения
├── logs/
│   └── ssl_loss.csv              # Лог лосса SSL-предобучения
├── SSL/
│   ├── augmentations.py          # Аугментации для SSL
│   ├── dataset.py                # SSL-датасет (две аугментации одного изображения)
│   ├── get_metrics.py            # Подсчёт метрик/визуализаций
│   ├── logger.py                 # CSV-логгер
│   ├── loss.py                   # NT-Xent loss
│   ├── model.py                  # SimCLR-модель
│   └── ssl_encoder.pth           # Сохранённый SSL encoder
└── hw1/
    ├── datasets.py               # Загрузка CIFAR-10 + ограничение доли разметки
    ├── model.py                  # EfficientNet-B0, загрузка SSL encoder, freeze
    ├── train.py                  # Train/eval функции
    ├── metrics.py                # Метрики классификации
    ├── main.py                   # Базовое supervised-обучение
    └── compare_feature_extractor.py # Сравнение baseline vs SSL extractor
```

## Подготовка окружения

Пример минимальной установки зависимостей:

```bash
pip install torch torchvision timm scikit-learn numpy matplotlib pandas
```

## 1. Self-supervised предобучение (SimCLR)

Предобучаем encoder на неразмеченных изображениях CIFAR-10 с контрастивной целью (NT-Xent).

```bash
python -m hw3.main \
  --epochs 100 \
  --batch_size 128 \
  --save_path hw3/SSL/ssl_encoder.pth
```

Результат:
- веса encoder сохраняются в `hw3/SSL/ssl_encoder.pth`;
- кривая SSL-лосса пишется в `hw3/logs/ssl_loss.csv`.

## 2. Базовый запуск supervised-модели (без сравнения)

Запуск обычного обучения классификатора на CIFAR-10:

```bash
python -m hw3.hw1.main
```

Этот сценарий обучает модель и выводит основные метрики на валидации/тесте.

## 3. Сравнение baseline и SSL feature extractor

Скрипт сравнивает два режима на одной и той же доле размеченной выборки:
- `baseline` — обучение модели с нуля;
- `feature_extractor` — загрузка `ssl_encoder.pth` и заморозка extractor (обучается в основном классификатор).

Примеры запуска:

```bash
python -m hw3.hw1.compare_feature_extractor --labeled-percent 100 --epochs 8
python -m hw3.hw1.compare_feature_extractor --labeled-percent 50 --epochs 8
python -m hw3.hw1.compare_feature_extractor --labeled-percent 10 --epochs 8 \
  --ssl-encoder-path hw3/SSL/ssl_encoder.pth
```

Полезные аргументы:
- `--labeled-percent` — доля размеченных данных (например, `100`, `50`, `10`);
- `--epochs` — число эпох;
- `--batch-size` — размер батча;
- `--lr` — learning rate;
- `--ssl-encoder-path` — путь к SSL-весам.

## 4. Результаты экспериментов

| Доля размеченных данных | Режим | Max Val Acc | Max Val Macro-F1 | Epoch до плато |
|---|---|---:|---:|---:|
| 10% | baseline | 43.41% | 0.4192 | 8 |
| 10% | feature_extractor | **67.76%** | **0.6765** | 8 |
| 50% | baseline | 68.83% | 0.6857 | 8 |
| 50% | feature_extractor | **71.28%** | **0.7107** | 8 |
| 100% | baseline | **77.77%** | **0.7774** | 8 |
| 100% | feature_extractor | 72.39% | 0.7228 | 7 |

Дополнительно по динамике (первая → последняя эпоха):

- **10% разметки**
  - baseline: Acc `27.26% → 43.41%`, F1 `0.2400 → 0.4192`;
  - feature extractor: Acc `46.00% → 67.74%`, F1 `0.4477 → 0.6749`.
- **50% разметки**
  - baseline: Acc `38.43% → 68.83%`, F1 `0.3693 → 0.6857`;
  - feature extractor: Acc `66.32% → 70.97%`, F1 `0.6616 → 0.7067`.
- **100% разметки**
  - baseline: Acc `47.54% → 77.77%`, F1 `0.4715 → 0.7774`;
  - feature extractor: Acc `68.26% → 72.39%`, F1 `0.6813 → 0.7228`.

### Выводы

1. **SSL особенно полезен при малом количестве разметки**: на 10% данных feature extractor даёт большой прирост по качеству относительно baseline.
2. **На 50% разметки SSL всё ещё выигрывает**, но отрыв уже заметно меньше.
3. **На 100% разметки baseline лучше по финальному качеству**, т.к. обучение всей модели с нуля на полном датасете превосходит замороженный extractor.
4. По критерию плато различия небольшие: в большинстве запусков обе конфигурации выходят на плато примерно к 8-й эпохе.
