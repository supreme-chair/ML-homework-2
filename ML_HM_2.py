
import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")




@app.cell
def _():
    import os
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from joblib import dump, load

    RF_PATH = "pc_price_model_rf.joblib"
    LR_PATH = "pc_price_model_lr.joblib"
    META_PATH = "pc_price_model_meta.joblib"

    model_rf = None
    model_lr = None
    X_columns = None
    cat_cols = None
    num_cols = None


    if os.path.exists(RF_PATH) and os.path.exists(LR_PATH) and os.path.exists(META_PATH):
        model_rf = load(RF_PATH)
        model_lr = load(LR_PATH)
        meta = load(META_PATH)
        X_columns = meta["X_columns"]
        cat_cols = meta["cat_cols"]
        num_cols = meta["num_cols"]
        print(">> RF and LR models downloaded from files")
    else:
        print(">> No model files found, we are training from scratch...")

        df = pd.read_csv("computer_prices_clean.csv")
        y = df["price"]
        X = df.drop(columns=["price"])

        cat_cols = [
            "device_type", "brand", "model", "os", "form_factor",
            "cpu_brand", "cpu_model", "gpu_brand", "gpu_model",
            "storage_type", "display_type", "resolution",
            "wifi", "bluetooth",
        ]
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        model_rf = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("rf", RandomForestRegressor(
                    n_estimators=20,
                    max_depth=17,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        )
        model_lr = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("lr", LinearRegression())
            ]
        )


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_rf.fit(X_train, y_train)
        model_lr.fit(X_train, y_train)


        y_pred_rf = model_rf.predict(X_test)
        y_pred_lr = model_lr.predict(X_test)

        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = sqrt(mse_rf)

        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = sqrt(mse_lr)

        print(f">> RandomForest  MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
        print(f">> LinearRegress MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")


        model_rf.fit(X, y)
        model_lr.fit(X, y)

        X_columns = X.columns.tolist()

        meta = {
            "X_columns": X_columns,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
        }
        dump(model_rf, RF_PATH)
        dump(model_lr, LR_PATH)
        dump(meta, META_PATH)
        print(">> Both models are trained on the entire dataset and saved to files")

    return model_rf, model_lr, X_columns, cat_cols, num_cols


@app.cell
def _():
    import marimo as mo
    return mo



@app.cell
def _(mo):
    header = mo.vstack(
        [
            mo.md("# CustomPC Pricing Dashboard"),
            mo.md(
                "Set up the configuration and get **the predicted price** "
                "taking into account the key parameters of the device."
            ),
        ]
    )
    header
    return header



@app.cell
def _(mo):

    device_type = mo.ui.dropdown(
        options=["Desktop", "Laptop"],
        value="Desktop",
        label="Device Type",
    )

    brand = mo.ui.dropdown(
        options=["Samsung", "Dell", "Apple", "HP", "Lenovo", "MSI", "Razer"],
        value="Dell",
        label="Device Brand",
    )

    storage_type = mo.ui.dropdown(
        options=["HDD", "SSD", "NVMe", "Hybrid"],
        value="SSD",
        label="Storage Type",
    )

    form_factor = mo.ui.dropdown(
        options=["Mainstream", "Ultrabook", "Gaming", "Workstation"],
        value="Mainstream",
        label="Form Factor",
    )


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
        label="Storage (GB)",
    )

    display_size_in = mo.ui.slider(
        start=13,
        stop=34,
        value=24,
        step=1,
        show_value=True,
        label="Display Size (inches)",
    )

    release_year = mo.ui.slider(
        start=2018,
        stop=2025,
        value=2023,
        step=1,
        show_value=True,
        label="Release Year",
    )


    col_left = mo.vstack(
        [
            mo.md("### Basic parameters"),
            device_type,
            brand,
            form_factor,
            display_size_in,
            release_year,
        ]
    )

    col_right = mo.vstack(
        [
            mo.md("### Performance and Storage"),
            cpu_tier,
            gpu_tier,
            ram_gb,
            storage_type,
            storage_gb,
        ]
    )

    ui = mo.vstack(
        [
            mo.md("## Configuration Settings"),
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


@app.cell
def _(
    mo,
    model_rf,
    model_lr,
    X_columns,
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
    import matplotlib.pyplot as plt


    row = {
        "device_type": device_type.value,
        "brand": brand.value,
        "model": "CustomUserBuild",
        "os": "Windows",
        "form_factor": form_factor.value,

        "cpu_brand": "Intel",
        "cpu_model": f"Tier{cpu_tier.value}",
        "cpu_tier": cpu_tier.value,
        "cpu_cores": 8,
        "cpu_threads": 16,
        "cpu_base_ghz": 2.5,
        "cpu_boost_ghz": 3.5,

        "gpu_brand": "NVIDIA",
        "gpu_model": f"Tier{gpu_tier.value}",
        "gpu_tier": gpu_tier.value,
        "vram_gb": 8,

        "ram_gb": ram_gb.value,

        "storage_type": storage_type.value,
        "storage_gb": storage_gb.value,
        "storage_drive_count": 1,

        "display_type": "LED",
        "display_size_in": display_size_in.value,
        "resolution": "1920x1080",
        "refresh_hz": 60,

        "battery_wh": 50,
        "charger_watts": 100,
        "psu_watts": 650,
        "wifi": "Wi-Fi 6",
        "bluetooth": "5.2",
        "weight_kg": 2.0,
        "warranty_months": 24,

        "release_year": release_year.value,
    }

    df_row = pd.DataFrame([row])

    for col in X_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[X_columns]

    price_rf = float(model_rf.predict(df_row)[0])
    price_lr = float(model_lr.predict(df_row)[0])

    card_rf = mo.vstack(
        [
            mo.md("### Random Forest"),
            mo.md(
                f"<div style='font-size:1.6rem; font-weight:700;'>~ {price_rf:,.2f} $</div>"
            ),
            mo.md(
                "_The nonlinear model, resistant to outliers; captures the interaction of objects well._"
            ),
        ],
        align="start",
    )

    card_lr = mo.vstack(
        [
            mo.md("### Linear Regression"),
            mo.md(
                f"<div style='font-size:1.6rem; font-weight:700;'>~ {price_lr:,.2f} $</div>"
            ),
            mo.md(
                "_The linear model with a clear interpretation; fast and simple._"
            ),
        ],
        align="start",
    )

    fig, ax = plt.subplots(figsize=(4.5, 3))

    labels = ["Random Forest", "Linear Regression"]
    values = [price_rf, price_lr]

    bars = ax.bar(labels, values)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Predicted price, $")
    ax.set_title("Comparison of model predictions", pad=8)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    details = mo.md(
        f"""
**Configuration:**

- Device Type: **{device_type.value}**
- Device Brand: **{brand.value}**
- Forv Factor: **{form_factor.value}**
- CPU tier: **{cpu_tier.value}**
- GPU tier: **{gpu_tier.value}**
- RAM: **{ram_gb.value} GB**
- Storage: **{storage_gb.value} GB ({storage_type.value})**
- Display Size: **{display_size_in.value}"**
- Realease Year: **{release_year.value}**
"""
    )

    layout = mo.vstack(
        [
            mo.md("## Price forecast (two models)"),
            mo.hstack([card_rf, card_lr], align="start", justify="space-between"),
            mo.md("### Visual comparison"),
            mo.hstack([fig], justify="center"),
            mo.md("---"),
            details,
        ]
    )

    layout


if __name__ == "__main__":
    app.run()
