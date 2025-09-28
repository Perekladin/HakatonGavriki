from road_tracks_full import process_hdf5_full

result = process_hdf5_full(
    file_path="Krasny_Yar_2_25_09_23_04_10_statistics.hdf5",
    freq_method="max",
    gaussian_sigma=(1.2, 1.2),
    median_size=3,
    threshold_abs=0.12,
    min_component_size=30,
    cross_section_positions=[0,200,400,600,800,1000,1200,1400,1600,2000,2500],
    save_visualization_path="tracks_overlay.png",
    save_speedmap_path="speed_map.png",
    output_json_path="tracks_full.json"
)

print("Готово! Найдено треков:", result["n_traces"])
print("Подсчет на сечениях:", result["cross_section_counts"])
print("Количество замедленных участков:", len(result["slow_sections"]))
