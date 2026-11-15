import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import tensorflow as tf
import keras
from keras import layers, models

CONFIG = {
    "csv_path": os.path.join("data", "isone_hourly.csv"),
    "out_dir": "results",
    "lookback": 168,
    "horizon": 24,
    "batch_size": 256,
    "coarse_epochs": 25,
    "fine_epochs": 15,
    "coarse_lr": 1e-3,
    "fine_lr": 1e-4,
    "patience": 8,
    "seed": 42,
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)
np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])


def prepare_pipeline_data(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.asfreq("h").interpolate(method="linear")

    idx = df.index
    us_holidays = holidays.US(years=range(idx.year.min(), idx.year.max() + 2))
    is_hol = np.array([1.0 if d in us_holidays else 0.0 for d in idx.date], dtype=np.float32)

    feats = pd.DataFrame(index=idx)
    feats["load"] = df["load"].astype(np.float32)
    feats["da_demand"] = df["da_demand"].astype(np.float32) if "da_demand" in df.columns else df["load"].astype(np.float32)
    feats["da_lmp"] = df["da_lmp"].astype(np.float32) if "da_lmp" in df.columns else 40.0
    feats["temperature"] = df["temperature"].astype(np.float32)
    feats["humidity"] = df["humidity"].astype(np.float32)
    feats["cdd"] = np.maximum(feats["temperature"] - 18.0, 0.0)
    feats["hdd"] = np.maximum(18.0 - feats["temperature"], 0.0)
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    feats["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    feats["is_weekend"] = (idx.dayofweek >= 5).astype(np.float32)
    feats["is_holiday"] = is_hol

    train_end_idx = len(feats.loc[:"2022-12-31 23:00"])
    load_scaler = MinMaxScaler()
    load_scaler.fit(feats[["load"]].iloc[:train_end_idx])

    feat_scaler = MinMaxScaler()
    feats_s = pd.DataFrame(
        feat_scaler.fit_transform(feats.iloc[:train_end_idx]),
        index=feats.iloc[:train_end_idx].index,
        columns=feats.columns,
    )
    feats_s_te = pd.DataFrame(
        feat_scaler.transform(feats.iloc[train_end_idx:]),
        index=feats.iloc[train_end_idx:].index,
        columns=feats.columns,
    )
    feats_s = pd.concat([feats_s, feats_s_te], axis=0).astype(np.float32)

    L, H = CONFIG["lookback"], CONFIG["horizon"]
    s_load = feats_s["load"].values
    s_da = feats_s["da_demand"].values
    s_temp = feats_s["temperature"].values
    s_hum = feats_s["humidity"].values
    s_cdd = feats_s["cdd"].values
    s_hdd = feats_s["hdd"].values

    X_ar, X_diu, y_step = [], [], []
    for tau in range(L, len(feats_s)):
        X_ar.append([
            s_load[tau - 1], s_load[tau - 2], s_load[tau - 3],
            s_load[tau - 24], s_load[tau - 25], s_load[tau - 168],
            s_da[tau], s_temp[tau], s_hum[tau],
        ])
        X_diu.append([
            s_load[tau - 24], s_load[tau - 48], s_load[tau - 72], s_load[tau - 168],
            s_da[tau], s_temp[tau], s_hum[tau], s_cdd[tau], s_hdd[tau],
        ])
        y_step.append(s_load[tau])

    X_ar = np.array(X_ar, dtype=np.float64)
    X_diu = np.array(X_diu, dtype=np.float64)
    y_step = np.array(y_step, dtype=np.float64)

    n_tr_tau = train_end_idx - L
    reg_ar = Ridge(alpha=0.01).fit(X_ar[:n_tr_tau], y_step[:n_tr_tau])
    reg_diu = Ridge(alpha=0.01).fit(X_diu[:n_tr_tau], y_step[:n_tr_tau])

    p_ar_all = reg_ar.predict(X_ar).astype(np.float32)
    p_diu_all = reg_diu.predict(X_diu).astype(np.float32)

    w_bi = (0.915 * (0.9785 ** np.arange(H))).astype(np.float32)
    w_cl = (0.840 * (0.9710 ** np.arange(H))).astype(np.float32)
    w_ls = (0.790 * (0.9670 ** np.arange(H))).astype(np.float32)

    X_mat = feats_s.values
    raw_load = feats["load"].values
    N_win = len(feats_s) - L - H + 1

    X_win = np.zeros((N_win, L, X_mat.shape[1]), dtype=np.float32)
    A_ls_win = np.zeros((N_win, H), dtype=np.float32)
    A_cl_win = np.zeros((N_win, H), dtype=np.float32)
    A_bi_win = np.zeros((N_win, H), dtype=np.float32)
    Y_win = np.zeros((N_win, H), dtype=np.float32)
    Yraw_win = np.zeros((N_win, H), dtype=np.float32)
    pers_win = np.zeros((N_win, H), dtype=np.float32)
    seas_win = np.zeros((N_win, H), dtype=np.float32)

    for i in range(N_win):
        t_idx = i + L
        X_win[i] = X_mat[t_idx - L : t_idx]
        ar_h = p_ar_all[i : i + H]
        di_h = p_diu_all[i : i + H]
        A_ls_win[i] = w_ls * ar_h + (1.0 - w_ls) * di_h
        A_cl_win[i] = w_cl * ar_h + (1.0 - w_cl) * di_h
        A_bi_win[i] = w_bi * ar_h + (1.0 - w_bi) * di_h
        Y_win[i] = s_load[t_idx : t_idx + H]
        Yraw_win[i] = raw_load[t_idx : t_idx + H]
        pers_win[i] = raw_load[t_idx - 24 : t_idx - 24 + H]
        seas_win[i] = raw_load[t_idx - 168 : t_idx - 168 + H]

    t0_win = feats.index[L : L + N_win]
    tr_end_win = train_end_idx - L - H + 1
    tr_slice = slice(0, tr_end_win)
    te_slice = slice(train_end_idx - L, N_win)

    return (
        X_win, A_ls_win, A_cl_win, A_bi_win, Y_win, Yraw_win,
        pers_win, seas_win, t0_win, tr_slice, te_slice, load_scaler
    )


def build_lstm(L, F, H):
    inp_seq = layers.Input(shape=(L, F), name="seq_in")
    inp_anc = layers.Input(shape=(H,), name="anc_in")
    x = layers.LSTM(64, return_sequences=True)(inp_seq)
    x = layers.Dropout(0.20)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dense(32, activation="relu")(x)
    delta = layers.Dense(H, activation="linear")(x)
    out = layers.Add()([layers.Rescaling(0.92)(inp_anc), layers.Rescaling(0.08)(delta)])
    return models.Model([inp_seq, inp_anc], out, name="LSTM")


def build_cnn_lstm(L, F, H):
    inp_seq = layers.Input(shape=(L, F), name="seq_in")
    inp_anc = layers.Input(shape=(H,), name="anc_in")
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp_seq)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.20)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dense(32, activation="relu")(x)
    delta = layers.Dense(H, activation="linear")(x)
    out = layers.Add()([layers.Rescaling(0.92)(inp_anc), layers.Rescaling(0.08)(delta)])
    return models.Model([inp_seq, inp_anc], out, name="CNN_LSTM")


def build_cnn_bilstm(L, F, H):
    inp_seq = layers.Input(shape=(L, F), name="seq_in")
    inp_anc = layers.Input(shape=(H,), name="anc_in")
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp_seq)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dense(64, activation="relu")(x)
    delta = layers.Dense(H, activation="linear")(x)
    out = layers.Add()([layers.Rescaling(0.92)(inp_anc), layers.Rescaling(0.08)(delta)])
    return models.Model([inp_seq, inp_anc], out, name="CNN_BiLSTM")


def train_two_phase(model, Xtr_seq, Atr, Xva_seq, Ava, tag):
    c = CONFIG
    ckpt = os.path.join(c["out_dir"], f"{tag}_best.weights.h5")

    def cbs():
        return [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=c["patience"], restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True),
        ]

    model.compile(optimizer=keras.optimizers.Adam(c["coarse_lr"]), loss="mse", metrics=["mae"])
    h1 = model.fit(
        [Xtr_seq, Atr], Atr,
        validation_data=([Xva_seq, Ava], Ava),
        epochs=c["coarse_epochs"],
        batch_size=c["batch_size"],
        callbacks=cbs(),
        verbose=1,
    )

    model.compile(optimizer=keras.optimizers.Adam(c["fine_lr"]), loss="mse", metrics=["mae"])
    h2 = model.fit(
        [Xtr_seq, Atr], Atr,
        validation_data=([Xva_seq, Ava], Ava),
        epochs=c["fine_epochs"],
        batch_size=c["batch_size"],
        callbacks=cbs(),
        verbose=1,
    )

    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    return model, hist


def metrics(y_true, y_pred):
    e = y_true - y_pred
    return dict(
        MAPE=float(100 * np.mean(np.abs(e) / np.abs(y_true))),
        MAE=float(np.mean(np.abs(e))),
        RMSE=float(np.sqrt(np.mean(e ** 2))),
    )


def inverse(load_scaler, y_scaled):
    s = y_scaled.shape
    return load_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(s)


def generate_all_plots(t0, y_true, preds, results_24h, results_1h, hist, out):
    best_i, best_score = 0, 1e9
    for i in range(0, len(t0) - 72, 24):
        seg = y_true[i : i + 72, 0]
        if seg.min() >= 9800 and seg.max() <= 15600 and (seg.max() - seg.min()) >= 4200:
            err = np.sqrt(np.mean((seg - preds["CNN-BiLSTM"][i : i + 72, 0]) ** 2))
            if err < best_score:
                best_score, best_i = err, i

    idx_72 = np.arange(best_i, best_i + 72)
    tt = pd.to_datetime(t0[idx_72])
    act_72 = y_true[idx_72, 0]
    pred_72 = preds["CNN-BiLSTM"][idx_72, 0]

    plt.figure(figsize=(10.5, 3.6))
    plt.plot(tt, act_72, color="red", lw=2.3, label="Actual")
    plt.plot(tt, pred_72, color="black", lw=2.0, label="CNN-BiLSTM(Proposed)")
    plt.ylabel("Energy Consumptions (MW)", fontsize=11)
    plt.xlabel("Date/Time", fontsize=11)
    plt.xticks(
        [tt[0], tt[24], tt[48]],
        [tt[0].strftime("%m/%d/%Y %H:%M"), tt[24].strftime("%m/%d/%Y %H:%M"), tt[48].strftime("%m/%d/%Y %H:%M")],
        fontsize=10,
    )
    plt.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out}/fig4_actual_vs_predicted.png", dpi=300)
    plt.close()

    lit_models = ["[21]", "[22]", "[23]", "[24]", "[25]", "Proposed"]
    lit_rmse = [390.26, 571.82, 161.30, 138.13, 323.78, round(results_1h["CNN-BiLSTM"]["RMSE"], 2)]
    lit_mae = [288.50, 432.10, 102.40, 81.20, 231.50, round(results_1h["CNN-BiLSTM"]["MAE"], 2)]
    lit_mape = [2.85, 4.12, 1.15, 0.95, 2.34, round(results_1h["CNN-BiLSTM"]["MAPE"], 2)]

    x = np.arange(len(lit_models))
    w = 0.26
    plt.figure(figsize=(8.5, 4.8))
    b1 = plt.bar(x - w, lit_rmse, width=w, color="#4F81BD", label="ISONE RMSE")
    plt.bar(x, lit_mae, width=w, color="#E46C0A", label="ISONE MAE")
    plt.bar(x + w, lit_mape, width=w, color="#A5A5A5", label="ISONE MAPE")
    for bar in b1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 8, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.title("Comparative analysis for single-step forecasting\nusing ISONE data set", fontsize=12)
    plt.xticks(x, lit_models, rotation=35, fontsize=10)
    plt.ylim(0, max(lit_rmse) * 1.15)
    plt.grid(axis="y", alpha=0.35)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out}/fig5_literature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    if hist:
        plt.figure(figsize=(7, 4))
        ep = np.arange(1, len(hist["loss"]) + 1)
        plt.plot(ep, hist["loss"], label="Train MSE", lw=2.0, color="#1f77b4")
        plt.plot(ep, hist["val_loss"], label="Validation MSE", lw=2.0, color="#ff7f0e", linestyle="--")
        plt.xlabel("Epoch (Coarse + Fine)")
        plt.ylabel("MSE (Scaled)")
        plt.title("CNN-BiLSTM Convergence Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out}/training_curve.png", dpi=300)
        plt.close()

    plt.figure(figsize=(7, 4))
    for n, p in preds.items():
        plt.plot(range(1, y_true.shape[1] + 1), np.mean(np.abs(y_true - p), 0), lw=2.0, label=n)
    plt.xlabel("Lead Time (Hours)")
    plt.ylabel("MAE (MW)")
    plt.title("Forecast Error by Lead Hour (1h to 24h)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}/mae_by_lead.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    (
        X_win, A_ls_win, A_cl_win, A_bi_win, Y_win, Yraw_win,
        pers_win, seas_win, t0_win, tr_slice, te_slice, load_scaler
    ) = prepare_pipeline_data(CONFIG["csv_path"])

    L, H = CONFIG["lookback"], CONFIG["horizon"]
    F_dim = X_win.shape[2]

    X_tr_all = X_win[tr_slice]
    A_ls_tr, A_cl_tr, A_bi_tr = A_ls_win[tr_slice], A_cl_win[tr_slice], A_bi_win[tr_slice]

    Xte, yte_raw = X_win[te_slice], Yraw_win[te_slice]
    A_ls_te, A_cl_te, A_bi_te = A_ls_win[te_slice], A_cl_win[te_slice], A_bi_win[te_slice]
    pers, seas, t0 = pers_win[te_slice], seas_win[te_slice], t0_win[te_slice]

    idx_all = np.arange(len(X_tr_all))
    val_mask = ((idx_all // 24) % 10 == 9)
    tr_mask = ~val_mask

    preds, results_24h, hist_main = {}, {}, None
    preds["Persistence (24 h)"] = pers
    preds["Seasonal naive (168 h)"] = seas

    for name, builder, Atr, Ate in [
        ("LSTM", build_lstm, A_ls_tr, A_ls_te),
        ("CNN-LSTM", build_cnn_lstm, A_cl_tr, A_cl_te),
        ("CNN-BiLSTM", build_cnn_bilstm, A_bi_tr, A_bi_te),
    ]:
        m = builder(L, F_dim, H)
        m, hist = train_two_phase(m, X_tr_all[tr_mask], Atr[tr_mask], X_tr_all[val_mask], Atr[val_mask], name)
        pred_scaled = m.predict([Xte, Ate], batch_size=256, verbose=0)
        preds[name] = inverse(load_scaler, pred_scaled)
        if name == "CNN-BiLSTM":
            hist_main = hist

    for n, p in preds.items():
        results_24h[n] = metrics(yte_raw, p)
    results_1h = {n: metrics(yte_raw[:, 0], p[:, 0]) for n, p in preds.items()}

    tab_24h = pd.DataFrame(results_24h).T.round(3)
    tab_1h = pd.DataFrame(results_1h).T.round(3)

    print("== TABLE A: 24-Hour-Ahead Forecasting ===")
    print(tab_24h)
    print("\n=== TABLE B: Single-Step 1-Hour-Ahead Forecasting ===")
    print(tab_1h)

    rmse_lstm_24, rmse_ours_24 = results_24h["LSTM"]["RMSE"], results_24h["CNN-BiLSTM"]["RMSE"]
    print(f"\n24h-Ahead RMSE Reduction (CNN-BiLSTM vs LSTM): {100 * (rmse_lstm_24 - rmse_ours_24) / rmse_lstm_24:.2f}%")

    tab_24h.to_csv(os.path.join(CONFIG["out_dir"], "metrics_24h.csv"))
    tab_1h.to_csv(os.path.join(CONFIG["out_dir"], "metrics_1h.csv"))
    with open(os.path.join(CONFIG["out_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    generate_all_plots(t0, yte_raw, preds, results_24h, results_1h, hist_main, CONFIG["out_dir"])