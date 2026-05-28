# DrumGuard – Setup & Deployment Guide

## Files

| File | Purpose |
|---|---|
| `drumguard_firmware.ino` | ESP32 Arduino firmware |
| `drumguard_simulate.py` | Offline validation against the Mendeley dataset |
| `drumguard_monitor.py`  | Real-time MQTT terminal dashboard |

---

## Hardware Wiring

```
ESP32 GPIO21 (SDA) ──────┬──── ADXL345 SDA
ESP32 GPIO22 (SCL) ──────┼──── ADXL345 SCL
                          ├──── INA219  SDA
                          └──── INA219  SCL

ESP32 3.3V ──────────────┬──── ADXL345 VCC
                          └──── INA219  VCC

ESP32 GND  ──────────────┬──── ADXL345 GND
                          └──── INA219  GND

INA219 V+  ────────────────── Motor supply rail
INA219 V-  ────────────────── Motor return rail
(Use a 0.1Ω shunt resistor between V+ and V- if not built-in)
```

**ADXL345 placement**: Stick to the printer chassis with double-sided foam tape,
close to the fuser assembly.

---

## ESP32 Firmware Setup

1. Install [Arduino IDE](https://www.arduino.cc/en/software) ≥ 2.x
2. Add ESP32 board support:  
   `File → Preferences → Additional Boards URLs`:  
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Install libraries (Sketch → Manage Libraries):
   - `Adafruit ADXL345`
   - `Adafruit INA219`
   - `PubSubClient`
   - `ArduinoJson`
4. Open `drumguard_firmware.ino`, fill in `WIFI_SSID`, `WIFI_PASSWORD`, `DEVICE_ID`.
5. Flash to ESP32 at 115200 baud.

---

## Dataset Simulation

```bash
# 1. Download dataset from:
#    https://data.mendeley.com/datasets/vxkj334rzv/7
# 2. Extract to ./dataset/

pip install numpy pandas matplotlib tqdm

# Run with defaults (10 calibration files, σ=3, ring=100)
python drumguard_simulate.py --dataset_dir ./dataset

# Tune parameters
python drumguard_simulate.py \
  --dataset_dir ./dataset \
  --calibration_files 15 \
  --sigma 2.5 \
  --ring_size 150 \
  --anomaly_ratio 0.5
```

python drumguard_simulate.py --dataset_dir ./dataset --no_plot --calibration_files 12 --sigma 2.5 --ring_size 150 --anomaly_ratio 0.2

Expected output:
```
═══ Phase 1: Calibration (Welford's Algorithm) ═══
  Calibrated on 1,234,567 samples from 10 files.
  Vibration  μ=0.98432 g   σ=0.04201  → threshold=1.11035
  Current    μ=1.24561 A   σ=0.07832  → threshold=1.47057

═══ Phase 2: Inference ═══
  healthy     samples=  850,000  alerts=      420  alert_rate=0.0%
  inner       samples=  920,000  alerts=  880,300  alert_rate=95.7%
  outer       samples=  870,000  alerts=  831,000  alert_rate=95.5%

═══ Classification Metrics ═══
  Accuracy   : 96.8%
  Precision  : 98.1%
  Recall     : 95.6%
  F1-Score   : 96.8%
```

---

## Real-time Monitor

```bash
pip install paho-mqtt rich
python drumguard_monitor.py --broker broker.hivemq.com
```

---

## Algorithm Summary

```
Phase 1 – Calibration (first 48 h on new printer)
─────────────────────────────────────────────────
For each sample:
  vibState.update(vibRMS)
  curState.update(currentA)

After calibration:
  vib_threshold = vib_μ + 3·vib_σ
  cur_threshold = cur_μ + 3·cur_σ
  (saved to ESP32 NVS flash)

Phase 2 – Inference
────────────────────
For each sample:
  vib_score = vibRMS / vib_threshold      # > 1.0 = breach
  cur_score = currentA / cur_threshold
  anomaly_score = max(vib_score, cur_score)
  ring_buffer.push(anomaly_score)

  if ring_buffer.fraction_above_1.0 >= 0.6:
      publish MQTT alert
```

---

## MQTT Payload Examples

**drumguard/data**
```json
{
  "device_id": "printer_01",
  "vibration_rms": 0.82,
  "current_amps": 1.31,
  "anomaly_score": 0.74,
  "status": "OK",
  "vib_threshold": 1.11,
  "cur_threshold": 1.47,
  "calibrated": true
}
```

**drumguard/alert**
```json
{
  "device_id": "printer_01",
  "vibration_rms": 1.44,
  "current_amps": 1.89,
  "anomaly_score": 1.29,
  "status": "WARNING",
  "message": "Sustained anomaly detected. Fuser assembly may be degrading."
}
```
