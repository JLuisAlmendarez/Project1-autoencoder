import tensorflow as tf
import time
import numpy as np
import os
import json
import datetime
import GPUtil
import psutil


# ==============================================================================
# CALLBACKS & LOGGER & UTILS
# ==============================================================================

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size     = batch_size
        self.training_start = None

    def on_train_begin(self, logs=None):
        self.training_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_time      = time.time() - self.batch_start
        samples_per_sec = self.batch_size / batch_time
        batches_per_sec = 1.0 / batch_time
        print(
            f"[Timing] Batch {batch} | "
            f"time: {batch_time:.4f}s | "
            f"samples/sec: {samples_per_sec:.2f} | "
            f"batches/sec: {batches_per_sec:.2f}"
        )
        self._log_batch(batch, {
            "batch_time_s":      round(batch_time, 4),
            "samples_per_sec":   round(samples_per_sec, 2),
            "batches_per_sec":   round(batches_per_sec, 2),
        })

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        wall_clock = time.time() - self.training_start
        print(
            f"[Timing] Epoch {epoch+1} | "
            f"epoch time: {epoch_time:.2f}s | "
            f"wall-clock total: {wall_clock:.2f}s"
        )
        self._log_epoch(epoch, {
            "epoch_time_s":    round(epoch_time, 2),
            "wall_clock_s":    round(wall_clock, 2),
        })

    # helpers para escritura al log central
    def _log_batch(self, batch, data):
        if hasattr(self, '_logger'):
            self._logger.log_batch(batch, "timing", data)

    def _log_epoch(self, epoch, data):
        if hasattr(self, '_logger'):
            self._logger.log_epoch(epoch, "timing", data)


class MemoryCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        ram_mb = psutil.Process().memory_info().rss / 1e6
        try:
            info        = tf.config.experimental.get_memory_info('GPU:0')
            vram_cur_mb = info['current'] / 1e6
            vram_pek_mb = info['peak']    / 1e6
        except Exception:
            vram_cur_mb = vram_pek_mb = 0.0
        print(
            f"[Memory] Batch {batch} | "
            f"RAM: {ram_mb:.1f} MB | "
            f"VRAM current: {vram_cur_mb:.1f} MB | "
            f"VRAM peak: {vram_pek_mb:.1f} MB"
        )
        if hasattr(self, '_logger'):
            self._logger.log_batch(batch, "memory", {
                "ram_mb":       round(ram_mb, 1),
                "vram_cur_mb":  round(vram_cur_mb, 1),
                "vram_peak_mb": round(vram_pek_mb, 1),
            })


class PipelineCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size      = batch_size
        self._last_batch_end = None

    def on_train_batch_begin(self, batch, logs=None):
        now = time.time()
        if self._last_batch_end is not None:
            load_time = now - self._last_batch_end
            io_throughput = self.batch_size / load_time
            print(
                f"[Pipeline] Batch {batch} | "
                f"data load: {load_time:.4f}s | "
                f"I/O throughput: {io_throughput:.2f} samples/sec"
            )
            if hasattr(self, '_logger'):
                self._logger.log_batch(batch, "pipeline", {
                    "data_load_s":    round(load_time, 4),
                    "io_throughput":  round(io_throughput, 2),
                })
        self._compute_start = now

    def on_train_batch_end(self, batch, logs=None):
        self._last_batch_end = time.time()
        compute_time = self._last_batch_end - self._compute_start
        print(f"[Pipeline] Batch {batch} | compute time: {compute_time:.4f}s")
        if hasattr(self, '_logger'):
            self._logger.log_batch(batch, "pipeline_compute", {
                "compute_time_s": round(compute_time, 4),
            })


class EnergyCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, flops_per_image=None):
        super().__init__()
        self.batch_size      = batch_size
        self.flops_per_image = flops_per_image
        self.total_energy    = 0.0
        self.total_samples   = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start
        cpu_frac   = psutil.cpu_percent(interval=None) / 100.0
        try:
            info     = tf.config.experimental.get_memory_info('GPU:0')
            gpu_frac = info['current'] / (info['peak'] + 1e-8)
        except Exception:
            gpu_frac = 0.0

        energy_batch        = batch_time * (cpu_frac + gpu_frac)
        self.total_energy  += energy_batch
        self.total_samples += self.batch_size

        print(
            f"[Energy] Batch {batch} | "
            f"energy_rel: {energy_batch:.4f} | "
            f"cpu_frac: {cpu_frac:.2f} | "
            f"gpu_frac: {gpu_frac:.2f}"
        )
        if hasattr(self, '_logger'):
            self._logger.log_batch(batch, "energy", {
                "energy_rel": round(energy_batch, 4),
                "cpu_frac":   round(cpu_frac, 2),
                "gpu_frac":   round(gpu_frac, 2),
            })

    def on_epoch_end(self, epoch, logs=None):
        eff_samples = self.total_samples / (self.total_energy + 1e-8)
        msg = (
            f"[Energy] Epoch {epoch+1} | "
            f"total energy_rel: {self.total_energy:.4f} | "
            f"efficiency: {eff_samples:.2f} samples/energy_unit"
        )
        epoch_data = {
            "total_energy_rel":          round(self.total_energy, 4),
            "efficiency_samples_per_eu": round(eff_samples, 2),
        }
        if self.flops_per_image is not None:
            total_flops = self.total_samples * self.flops_per_image
            eff_flops   = total_flops / (self.total_energy + 1e-8)
            msg += f" | FLOPs/energy_unit: {eff_flops:.2e}"
            epoch_data["efficiency_flops_per_eu"] = float(f"{eff_flops:.2e}")
        print(msg)
        if hasattr(self, '_logger'):
            self._logger.log_epoch(epoch, "energy", epoch_data)
        self.total_energy  = 0.0
        self.total_samples = 0


class UtilizationCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        cpu  = psutil.cpu_percent()
        gpus = GPUtil.getGPUs()
        gpu  = gpus[0].load * 100 if gpus else 0.0
        print(f"[Utilization] Batch {batch} | CPU: {cpu:.1f}% | GPU: {gpu:.1f}%")
        if hasattr(self, '_logger'):
            self._logger.log_batch(batch, "utilization", {
                "cpu_pct": round(cpu, 1),
                "gpu_pct": round(gpu, 1),
            })

class TrainingLogger:
    """
    Escribe metrics.json y summary.json dentro de log_dir (mismo que TensorBoard).
    Estructura de metrics.json:
        {
          "epochs": {
            "0": {
              "batches": {
                "0": { "timing": {...}, "memory": {...}, ... },
                ...
              },
              "epoch_summary": { "timing": {...}, "energy": {...}, "losses": {...} }
            },
            ...
          }
        }
    """
    def __init__(self, log_dir, config):
        self.log_dir      = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics.json")
        self.summary_path = os.path.join(log_dir, "summary.json")
        self._data        = {"epochs": {}}
        self._current_epoch = 0

        # Escribir config inicial en summary
        self._summary = {
            "run_id":       os.path.basename(log_dir),
            "started_at":   datetime.datetime.now().isoformat(),
            "config":       config,
            "completed_at": None,
            "total_epochs": None,
            "wall_clock_s": None,
        }
        os.makedirs(log_dir, exist_ok=True)
        self._write_summary()

    def set_epoch(self, epoch):
        self._current_epoch = epoch
        epoch_key = str(epoch)
        if epoch_key not in self._data["epochs"]:
            self._data["epochs"][epoch_key] = {"batches": {}, "epoch_summary": {}}

    def log_batch(self, batch, category, data):
        epoch_key = str(self._current_epoch)
        batch_key = str(batch)
        if epoch_key not in self._data["epochs"]:
            self._data["epochs"][epoch_key] = {"batches": {}, "epoch_summary": {}}
        if batch_key not in self._data["epochs"][epoch_key]["batches"]:
            self._data["epochs"][epoch_key]["batches"][batch_key] = {}
        self._data["epochs"][epoch_key]["batches"][batch_key][category] = data
        self._write_metrics()

    def log_epoch(self, epoch, category, data):
        epoch_key = str(epoch)
        if epoch_key not in self._data["epochs"]:
            self._data["epochs"][epoch_key] = {"batches": {}, "epoch_summary": {}}
        self._data["epochs"][epoch_key]["epoch_summary"][category] = data
        self._write_metrics()

    def log_losses(self, epoch, logs):
        self.log_epoch(epoch, "losses", {
            k: round(float(v), 4) for k, v in (logs or {}).items()
        })

    def finalize(self, total_epochs, wall_clock_s):
        self._summary["completed_at"] = datetime.datetime.now().isoformat()
        self._summary["total_epochs"] = total_epochs
        self._summary["wall_clock_s"] = round(wall_clock_s, 2)
        self._write_summary()

    def _write_metrics(self):
        with open(self.metrics_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _write_summary(self):
        with open(self.summary_path, "w") as f:
            json.dump(self._summary, f, indent=2)


class LoggerBridgeCallback(tf.keras.callbacks.Callback):
    """
    Conecta el TrainingLogger con todos los callbacks y captura
    losses de Keras al final de cada epoch.
    """
    def __init__(self, logger, callbacks_to_bridge):
        super().__init__()
        self.logger   = logger
        self._cbs     = callbacks_to_bridge
        self._t_start = None

    def on_train_begin(self, logs=None):
        self._t_start = time.time()
        for cb in self._cbs:
            cb._logger = self.logger

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.set_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.logger.log_losses(epoch, logs)

    def on_train_end(self, logs=None):
        wall_clock = time.time() - self._t_start
        self.logger.finalize(
            total_epochs=self.params.get("epochs", "?"),
            wall_clock_s=wall_clock,
        )
        print(f"\n[Logger] metrics.json  → {self.logger.metrics_path}")
        print(f"[Logger] summary.json  → {self.logger.summary_path}")

def compute_static_flops(model, img_shape, batch_size):
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )
        dummy = tf.ones((1, *img_shape))
        _     = model(dummy, training=False)

        @tf.function
        def forward(x):
            return model(x, training=False)

        concrete  = forward.get_concrete_function(tf.TensorSpec([1, *img_shape], tf.float32))
        frozen    = convert_variables_to_constants_v2(concrete)
        graph_def = frozen.graph.as_graph_def()

        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts     = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            result   = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd="op", options=opts)

        flops_per_image = result.total_float_ops
        flops_per_batch = flops_per_image * batch_size
        print(
            f"[FLOPs] Por imagen: {flops_per_image:,} ({flops_per_image/1e6:.2f} MFLOPs) | "
            f"Por batch ({batch_size}): {flops_per_batch:,} ({flops_per_batch/1e9:.2f} GFLOPs)"
        )
        return flops_per_image, flops_per_batch

    except Exception as e:
        print(f"[FLOPs] No se pudo calcular: {e}")
        return None, None


# ==============================================================================
# Experimental Scheduler
# ==============================================================================
class AdaptiveLR(tf.keras.callbacks.Callback):
    """
    Scheduler adaptativo de tres fases basado en el valor Y la tendencia de la loss.

    Fases y condiciones:
    ┌─────────────────┬──────────────────────────────┬───────────────────┐
    │ Zona de loss    │ Tendencia (delta_mavg)        │ Fase / Accion     │
    ├─────────────────┼──────────────────────────────┼───────────────────┤
    │ > loss_high     │ bajando  (< -epsilon)         │ exploration / =   │
    │ > loss_high     │ estancada o subiendo          │ acceleration / *up│
    │ loss_low..high  │ cualquiera                    │ exploration / =   │
    │ < loss_low      │ bajando  (< -epsilon)         │ exploration / =   │
    │ < loss_low      │ estancada (>= -epsilon)       │ fine_tuning / *dn │
    │ cualquiera      │ divergiendo (> +diverge_thr)  │ divergence / *dn  │
    └─────────────────┴──────────────────────────────┴───────────────────┘

    La deteccion de divergencia tiene prioridad sobre las demas fases.

    Parametros:
        lr_max          : techo absoluto del LR
        lr_min          : piso absoluto del LR
        f_up            : factor multiplicativo de subida   (ej. 1.05 = +5%)
        f_down          : factor multiplicativo de bajada   (ej. 0.92 = -8%)
        loss_high       : umbral superior — calibrar segun tu loss real
        loss_low        : umbral inferior — calibrar segun convergencia esperada
        momentum_window : ventana de epochs para calcular delta medio
        epsilon         : cambio minimo absoluto para considerar que la loss baja
        diverge_thr     : delta positivo que indica divergencia sostenida
                          Por defecto = epsilon * 2 (sube el doble de lo que bajaria)
    """
    def __init__(self,
                 lr_max=1e-3,
                 lr_min=1e-6,
                 f_up=1.05,
                 f_down=0.92,
                 loss_high=500.0,
                 loss_low=100.0,
                 momentum_window=5,
                 epsilon=1.0,
                 diverge_thr=None):
        super().__init__()
        self.lr_max          = lr_max
        self.lr_min          = lr_min
        self.f_up            = f_up
        self.f_down          = f_down
        self.loss_high       = loss_high
        self.loss_low        = loss_low
        self.momentum_window = momentum_window
        self.epsilon         = epsilon
        self.diverge_thr     = diverge_thr if diverge_thr is not None else epsilon * 2
        self.loss_history    = []

    def _get_lr(self):
        return float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def _set_lr(self, new_lr):
        try:
            self.model.optimizer.learning_rate.assign(new_lr)
        except AttributeError:
            self.model.optimizer.learning_rate = new_lr

    def on_epoch_end(self, epoch, logs=None):
        current_loss = (logs or {}).get("loss")
        if current_loss is None:
            return

        # Ventana deslizante de historial
        self.loss_history.append(float(current_loss))
        if len(self.loss_history) > self.momentum_window:
            self.loss_history.pop(0)

        # Delta medio: negativo = bajando, positivo = subiendo o estancado
        delta_mavg = float(np.mean(np.diff(self.loss_history))) if len(self.loss_history) > 1 else 0.0

        lr = self._get_lr()

        # --- Deteccion de divergencia (prioridad maxima) ---
        if delta_mavg > self.diverge_thr:
            phase  = "divergence"
            new_lr = max(lr * self.f_down, self.lr_min)

        # --- Zona alta ---
        elif current_loss > self.loss_high:
            if delta_mavg < -self.epsilon:
                phase  = "exploration"   # bajando bien, no tocar
                new_lr = lr
            else:
                phase  = "acceleration"  # estancado en zona alta, empujar
                new_lr = min(lr * self.f_up, self.lr_max)

        # --- Zona media ---
        elif current_loss > self.loss_low:
            phase  = "exploration"       # progreso normal, no interferir
            new_lr = lr

        # --- Zona baja ---
        else:
            if delta_mavg < -self.epsilon:
                phase  = "exploration"   # sigue bajando, dejar correr
                new_lr = lr
            else:
                phase  = "fine_tuning"   # estancado cerca del minimo, afinar
                new_lr = max(lr * self.f_down, self.lr_min)

        self._set_lr(new_lr)

        print(
            f"[AdaptiveLR] Epoch {epoch+1} | "
            f"loss: {current_loss:.2f} | "
            f"delta_mavg: {delta_mavg:+.4f} | "
            f"phase: {phase:<12} | "
            f"lr: {new_lr:.2e}"
        )

        if hasattr(self, "_logger"):
            self._logger.log_epoch(epoch, "adaptive_lr", {
                "loss":        round(float(current_loss), 4),
                "delta_mavg":  round(delta_mavg, 4),
                "phase":       phase,
                "lr":          float(f"{new_lr:.2e}"),
            })
