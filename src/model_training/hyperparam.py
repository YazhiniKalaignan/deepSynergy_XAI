# %%
import itertools
import sys
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
ROOT_DIR = Path.cwd().parents[1]
sys.path.append(str(ROOT_DIR / "src" / "data_preprocessing"))
from normalize_fn import load
# %%
norm_options = ['norm', 'tanh', 'tanh_norm']
hidden_options = [
    [4096, 2048],
    [2048, 1024],
    [4096, 2048, 1024],
    [2048, 1024, 512],
    [1024, 1024]]
lr_options = [1e-2, 1e-3, 1e-4, 1e-5]
dropout_options = [(0, 0), (0.2, 0.5)]
hyperparameter_grid = list(itertools.product(
    norm_options, hidden_options, lr_options, dropout_options
))
best_val_loss = np.inf
best_params = None
best_epoch = None
# %%
def moving_average(x, n):
    return np.convolve(x, np.ones(n) / n, mode='valid')
# %%
for norm_type, hidden_layers, lr, (input_do, hidden_do) in hyperparameter_grid:
    X_tr_np, X_val_np, _, _, y_tr_np, y_val_np, _, _ = load(norm=norm_type)
    model = Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(Dense(
                units,
                input_shape=(X_tr_np.shape[1],),
                activation='relu',
                kernel_initializer='he_normal'
            ))
            if input_do > 0:
                model.add(Dropout(input_do))
        else:
            model.add(Dense(
                units,
                activation='relu',
                kernel_initializer='he_normal'
            ))
            if hidden_do > 0:
                model.add(Dropout(hidden_do))
    model.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
    model.compile(
        loss='mean_squared_error',
        optimizer=SGD(
            learning_rate=lr,
            momentum=0.5,
            clipnorm=1.0
        )
    )
    hist = model.fit(
        X_tr_np, y_tr_np,
        validation_data=(X_val_np, y_val_np),
        epochs=50,
        batch_size=64,
        shuffle=True,
        verbose=0
    )
    val_losses = np.array(hist.history['val_loss'])
    if np.isnan(val_losses).any():
        continue
    ma_losses = moving_average(val_losses, n=25)
    best_ma_loss = np.min(ma_losses)
    best_ma_epoch = np.argmin(ma_losses) + 25
    if best_ma_loss < best_val_loss:
        best_val_loss = best_ma_loss
        best_epoch = best_ma_epoch
        best_params = {
            "norm": norm_type,
            "hidden_layers": hidden_layers,
            "learning_rate": lr,
            "input_dropout": input_do,
            "hidden_dropout": hidden_do,
            "epochs": best_ma_epoch }
# %%
out_file = ROOT_DIR / "best_hyperparams.txt"
with open(out_file, "w") as f:
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"best_val_loss: {best_val_loss}\n")
