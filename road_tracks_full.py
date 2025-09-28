# Импорт необходимых библиотек
import h5py  # Для работы с HDF5 файлами
import numpy as np  # Для работы с массивами данных
from skimage import measure, morphology  # Для работы с изображениями и морфологических операций
from scipy.ndimage import gaussian_filter, median_filter  # Для фильтрации изображений
import json  # Для работы с JSON файлами
import matplotlib.pyplot as plt  # Для построения графиков

def process_hdf5_full(file_path,
                      freq_method="max",  # Метод агрегации по частотам
                      gaussian_sigma=(1.2, 1.2),  # Параметры сглаживания Гаусса
                      median_size=3,  # Размер медианного фильтра
                      threshold_abs=0.12,  # Абсолютный порог бинаризации
                      min_component_size=30,  # Минимальный размер компонента
                      cross_section_positions=None,  # Позиции для подсчета автомобилей
                      save_visualization_path="tracks_overlay.png",  # Путь для сохранения визуализации
                      save_speedmap_path="speed_map.png",  # Путь для сохранения карты скорости
                      output_json_path="tracks_full.json"):  # Путь для сохранения результатов
    """
    Обрабатывает HDF5 данные и строит треки автомобилей с вычислением скорости,
    транспортного потока и визуализацией замедленных участков.
    """

    # 1. Загрузка данных из HDF5 файла
    with h5py.File(file_path, "r") as f:
        statistics = f['statistics'][:]  # Чтение массива статистики

    # Определение размеров данных
    n_time, n_distance, n_freq = statistics.shape  # Количество временных отсчетов, дистанций и частот
    dt = 0.62  # Временной шаг между измерениями (в секундах)
    time_stamps = np.arange(n_time) * dt  # Создание временной шкалы

    # 2. Агрегация данных по частотам
    if freq_method == "max":
        signal_2d = statistics.max(axis=2)  # Берем максимум по частотной оси
    elif freq_method == "sum":
        signal_2d = statistics.sum(axis=2)  # Суммируем по частотам
    elif freq_method == "mean":
        signal_2d = statistics.mean(axis=2)  # Усредняем по частотам
    else:
        raise ValueError("Unknown freq_method")

    # 3. Сглаживание данных фильтрами
    signal_2d = gaussian_filter(signal_2d, sigma=gaussian_sigma)  # Сглаживание Гауссовым фильтром
    signal_2d = median_filter(signal_2d, size=(median_size, 1))  # Медианный фильтр по времени

    # 4. Бинаризация и очистка от шума
    threshold = signal_2d.max() * threshold_abs  # Вычисление порога бинаризации
    binary = signal_2d > threshold  # Создание бинарной маски
    binary = morphology.remove_small_objects(binary, min_size=min_component_size)  # Удаление мелких объектов

    # 5. Сегментация и анализ компонентов
    labeled = measure.label(binary)  # Разметка связанных компонентов
    regions = measure.regionprops(labeled)  # Получение свойств каждого компонента

    trace_list = []  # Список для хранения треков

    # Инициализация карт для расчета скорости
    speed_map = np.zeros(n_distance)  # Сумма скоростей в каждой точке
    count_map = np.zeros(n_distance)  # Количество измерений в каждой точке

    # Обработка каждого обнаруженного объекта
    for idx, region in enumerate(regions):
        coords = region.coords  # Получение координат пикселей компонента
        coords = coords[np.argsort(coords[:, 0])]  # Сортировка по времени (оси Y)
        points = []  # Точки текущего трека
        speeds = []  # Скорости между точками

        # Обработка каждой точки в компоненте
        for i in range(len(coords)):
            r, c = coords[i]  # Извлечение координат (время, позиция)
            t = float(time_stamps[r])  # Перевод во временную координату
            pos = int(c)  # Пространственная координата
            points.append({"time": t, "position": pos})  # Добавление точки в трек

            # Расчет скорости между точками
            if i > 0:
                dt_i = t - points[i-1]["time"]  # Временной интервал
                dx_i = pos - points[i-1]["position"]  # Пройденное расстояние
                speed = dx_i / dt_i if dt_i > 0 else 0  # Вычисление скорости
                speeds.append(speed)

                # Заполнение карты скорости
                pos_range = range(min(points[i-1]["position"], points[i]["position"]),
                                  max(points[i-1]["position"], points[i]["position"])+1)
                for p in pos_range:  # Для всех позиций между точками
                    if p < n_distance:  # Проверка границ
                        speed_map[p] += speed  # Добавление скорости
                        count_map[p] += 1  # Увеличение счетчика

        # Статистика по треку
        avg_speed = np.mean(speeds) if speeds else 0  # Средняя скорость
        trace_list.append({
            "id": idx,  # Идентификатор трека
            "points": points,  # Список точек
            "average_speed": avg_speed,  # Средняя скорость
            "length": len(points),  # Длина трека
            "max_speed": np.max(speeds) if speeds else 0  # Максимальная скорость
        })

    # Расчет средней скорости по всем трекам
    avg_speed_map = np.divide(speed_map, count_map, out=np.zeros_like(speed_map), where=count_map!=0)

    # Определение участков с замедлением
    median_speed = np.median([t["average_speed"] for t in trace_list if t["average_speed"] > 0]) if trace_list else 0
    slow_sections = [i for i, v in enumerate(avg_speed_map) if median_speed > 0 and v < 0.5 * median_speed]

    # Подсчет автомобилей на заданных сечениях
    cross_section_counts = {}
    if cross_section_positions:
        for pos in cross_section_positions:  # Для каждой контрольной точки
            # Подсчет треков, проходящих через точку
            count = sum(any(p["position"] == pos for p in t["points"]) for t in trace_list)
            cross_section_counts[pos] = count

    # Сохранение результатов в JSON
    output = {
        "trace_list": trace_list,  # Список треков
        "cross_section_counts": cross_section_counts,  # Подсчет на сечениях
        "n_traces": len(trace_list),  # Количество треков
        "slow_sections": slow_sections,  # Участки замедления
        "avg_speed_map": avg_speed_map.tolist()  # Карта средней скорости
    }

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)  # Запись с форматированием

    # Визуализация треков поверх исходных данных
    plt.figure(figsize=(14, 6))
    plt.imshow(signal_2d, aspect='auto', cmap='gray', origin='lower')  # Исходные данные
    for t in trace_list:  # Отрисовка каждого трека
        y = [p["time"] for p in t["points"]]
        x = [p["position"] for p in t["points"]]
        plt.plot(x, y, linewidth=1)
    plt.xlabel("Дистанция (м)")
    plt.ylabel("Время (с)")
    plt.title("Треки автомобилей")
    plt.savefig(save_visualization_path)
    plt.close()

    # Визуализация карты скорости
    plt.figure(figsize=(14, 4))
    plt.plot(avg_speed_map)  # График средней скорости
    plt.scatter(slow_sections, avg_speed_map[slow_sections], color='red', s=10, label='Замедление')  # Точки замедления
    plt.xlabel("Дистанция (м)")
    plt.ylabel("Средняя скорость")
    plt.title("Карта скорости и замедления на участке дороги")
    plt.legend()
    plt.savefig(save_speedmap_path)
    plt.close()

    return output  # Возврат результатов
