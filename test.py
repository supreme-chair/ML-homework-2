import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    # Тут пока просто заглушка, потом сюда можно будет засунуть загрузку датасета/модели
    import numpy as np
    import matplotlib.pyplot as plt

    def plot(x, y, figsize=(3, 3), lw=2):
        fig = plt.figure(figsize=figsize)
        for xx, yy in zip(x, y):
            plt.plot(xx, yy, lw=lw)
        return fig


@app.cell
def _():
    import marimo as mo
    return mo


# ШАПКА
@app.cell
def _(mo):
    header = mo.vstack(
        [
            mo.md("# CustomPC Pricing Dashboard"),
            mo.md(
                "Настройте конфигурацию и получите **прогнозируемую цену** "
                "с учётом ключевых параметров устройства."
            ),
        ]
    )
    header
    return header


# ЯЧЕЙКА 1: только UI, никаких .value
@app.cell
def _(mo):
    # Категориальные
    device_type = mo.ui.dropdown(
        options=["Desktop", "Laptop"],
        value="Desktop",
        label="Тип устройства (device_type)",
    )

    brand = mo.ui.dropdown(
        options=["Samsung", "Dell", "Apple", "HP", "Lenovo", "MSI", "Razer"],
        value="Dell",
        label="Бренд (brand)",
    )

    storage_type = mo.ui.dropdown(
        options=["HDD", "SSD", "NVMe", "Hybrid"],
        value="SSD",
        label="Тип накопителя (storage_type)",
    )

    form_factor = mo.ui.dropdown(
        options=["Mainstream", "Ultrabook", "Gaming", "Workstation"],
        value="Mainstream",
        label="Форм-фактор (form_factor)",
    )

    # Числовые
    cpu_tier = mo.ui.slider(
        start=1,
        stop=6,
        value=3,
        step=1,
        show_value=True,
        label="CPU tier (1–6)",
    )

    gpu_tier = mo.ui.slider(
        start=1,
        stop=6,
        value=3,
        step=1,
        show_value=True,
        label="GPU tier (1–6)",
    )

    ram_gb = mo.ui.slider(
        start=8,
        stop=128,
        value=16,
        step=8,
        show_value=True,
        label="RAM (GB)",
    )

    storage_gb = mo.ui.slider(
        start=256,
        stop=4096,
        value=512,
        step=256,
        show_value=True,
        label="Накопитель (GB)",
    )

    display_size_in = mo.ui.slider(
        start=13,
        stop=34,
        value=24,
        step=1,
        show_value=True,
        label="Диагональ дисплея (дюймы)",
    )

    release_year = mo.ui.slider(
        start=2018,
        stop=2025,
        value=2023,
        step=1,
        show_value=True,
        label="Год выпуска",
    )

    # Разбиваем параметры на две логичные колонки
    col_left = mo.vstack(
        [
            mo.md("### Основные параметры"),
            device_type,
            brand,
            form_factor,
            display_size_in,
            release_year,
        ]
    )

    col_right = mo.vstack(
        [
            mo.md("### Производительность и хранилище"),
            cpu_tier,
            gpu_tier,
            ram_gb,
            storage_type,
            storage_gb,
        ]
    )

    ui = mo.vstack(
        [
            mo.md("## Параметры конфигурации"),
            mo.hstack([col_left, col_right], align="start"),
        ]
    )

    ui

    return (
        device_type,
        brand,
        form_factor,
        storage_type,
        cpu_tier,
        gpu_tier,
        ram_gb,
        storage_gb,
        display_size_in,
        release_year,
    )


# ЯЧЕЙКА 2: читаем .value и считаем "цену"
@app.cell
def _(
    mo,
    device_type,
    brand,
    form_factor,
    storage_type,
    cpu_tier,
    gpu_tier,
    ram_gb,
    storage_gb,
    display_size_in,
    release_year,
):
    # базовая цена
    price = 500.0

    # device_type: ноут обычно дороже
    if device_type.value == "Laptop":
        price += 250

    # CPU tier: сильное влияние
    price += (cpu_tier.value - 1) * 180

    # GPU tier: ещё сильнее влияет для игровых / рабочих станций
    gpu_multiplier = 220
    if form_factor.value in ("Gaming", "Workstation"):
        gpu_multiplier = 260
    price += (gpu_tier.value - 1) * gpu_multiplier

    # RAM: до 32 растём сильно, дальше слабее
    if ram_gb.value <= 32:
        price += (ram_gb.value - 8) * 5
    else:
        price += (32 - 8) * 5 + (ram_gb.value - 32) * 2

    # Storage size: линейный вклад
    price += (storage_gb.value - 256) * 0.1

    # Storage type: HDD < SSD < NVMe < Hybrid (условно)
    if storage_type.value == "HDD":
        price += 0
    elif storage_type.value == "SSD":
        price += 80
    elif storage_type.value == "NVMe":
        price += 150
    elif storage_type.value == "Hybrid":
        price += 120

    # Display size: больше экран — дороже
    price += (display_size_in.value - 13) * 15

    # Release year: свежее — дороже, но старым делаем скидку
    age = 2025 - release_year.value
    if age <= 1:
        price += 200
    elif age <= 3:
        price += 100
    else:
        price -= age * 50  # старым снижаем цену

    # Бренд: Apple / Razer — премиум
    if brand.value == "Apple":
        price *= 1.25
    elif brand.value == "Razer":
        price *= 1.15
    elif brand.value in ("MSI", "Samsung"):
        price *= 1.05

    # Форм-фактор: Gaming / Workstation чуть дороже
    if form_factor.value == "Gaming":
        price += 200
    elif form_factor.value == "Workstation":
        price += 300

    # Минимальная защита от отрицательной цены
    price = max(price, 300)

    summary = mo.vstack(
        [
            mo.md("## Итоговая оценка конфигурации"),
            mo.md(
                f"<div style='font-size: 2rem; font-weight: 700;'>"
                f"~ {price:,.2f} у.е."
                f"</div>"
            ),
            mo.md(
                """
---

### Краткое резюме конфигурации
"""
            ),
            mo.md(
                f"""
- **Тип устройства:** {device_type.value}  
- **Бренд:** {brand.value}  
- **Форм-фактор:** {form_factor.value}  
- **CPU tier:** {cpu_tier.value}  
- **GPU tier:** {gpu_tier.value}  
- **RAM:** {ram_gb.value} GB  
- **Накопитель:** {storage_gb.value} GB {storage_type.value}  
- **Диагональ:** {display_size_in.value}"  
- **Год выпуска:** {release_year.value}  
"""
            ),
        ]
    )

    summary


if __name__ == "__main__":
    app.run()
