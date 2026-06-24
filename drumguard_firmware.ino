/**
 * DrumGuard – Predictive Printer Failure Detection System
 * ESP32 Firmware
 *
 * Hardware:
 *   - ESP32 (any variant)
 *   - ADXL345 (I2C, vibration)
 *   - INA219  (I2C, motor current)
 *
 * Algorithm:
 *   Phase 1 – Welford's Online Algorithm for dynamic baselining (48 h)
 *   Phase 2 – Ring Buffer (100 samples) rolling variance vs μ + 3σ threshold
 *
 * Dataset reference:
 *   Vibration and Motor Current Dataset of Rolling Element Bearing
 *   Under Varying Speed Conditions for Fault Diagnosis
 *   https://data.mendeley.com/datasets/vxkj334rzv/7
 *
 * MQTT topics:
 *   drumguard/data   – telemetry (JSON)
 *   drumguard/status – health string
 *   drumguard/alert  – anomaly payload (JSON)
 */

// ─────────────────────────── Includes ───────────────────────────────────────
#include <driver/i2s.h>
#include "arduinoFFT.h"
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_INA219.h>
#include <math.h>
#include <Preferences.h>   // NVS – persist baseline across reboots

// ─────────────────────────── Config ────────────────────────────────────────
#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 14
#define I2S_PORT I2S_NUM_0

#define I2S_SAMPLE_RATE     16000
#define FFT_SAMPLES         1024  // Must be power of 2
#define NOISE_FLOOR_HZ      4000  // Ignore < 4kHz
// ─────────────────────────── User config ────────────────────────────────────
#define WIFI_SSID        "YOUR_WIFI_SSID"
#define WIFI_PASSWORD    "YOUR_WIFI_PASSWORD"
#define MQTT_BROKER      "broker.hivemq.com"
#define MQTT_PORT        1883
#define DEVICE_ID        "printer_01"

// Calibration window: 48 hours = 172 800 seconds.
// At 10 Hz sampling that is 1 728 000 samples.
// We store only running statistics (Welford), so RAM usage is O(1).
#define CALIBRATION_DURATION_S  (48UL * 3600UL)
#define SAMPLE_INTERVAL_MS      100      // 10 Hz
#define RING_BUFFER_SIZE        100      // rolling window for inference
#define SIGMA_MULTIPLIER        3.0f     // threshold = μ + k·σ
#define ANOMALY_CONFIRM_RATIO   0.6f     // fraction of ring buffer > threshold

// ─────────────────────────── Hardware objects ───────────────────────────────
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
Adafruit_INA219           ina219;
WiFiClient                wifiClient;
PubSubClient              mqtt(wifiClient);
Preferences               prefs;
double vReal[FFT_SAMPLES];
double vImag[FFT_SAMPLES];
arduinoFFT FFT = arduinoFFT(vReal, vImag, FFT_SAMPLES, I2S_SAMPLE_RATE);

// ─────────────────────────── Welford state ──────────────────────────────────
struct WelfordState {
    double  mean;
    double  M2;          // sum of squared deviations
    uint64_t count;
    double  stddev() const {
        return (count < 2) ? 0.0 : sqrt(M2 / (count - 1));
    }
};

// Shared state between RTOS cores
volatile float latestHighBandEnergy = 0.0f;
// Add to existing Welford states
WelfordState audioState;
double audioThreshold = 0.0;
WelfordState vibState;
WelfordState curState;

// ─────────────────────────── Ring buffer ────────────────────────────────────
// Stores the combined anomaly score of each recent sample.
float  ringBuf[RING_BUFFER_SIZE];
int    ringHead  = 0;
bool   ringFull  = false;

// ─────────────────────────── Runtime state ──────────────────────────────────
bool     calibrated      = false;
uint32_t calibStartMs    = 0;
double   vibThreshold    = 0.0;
double   curThreshold    = 0.0;
uint32_t lastSampleMs    = 0;
uint32_t lastMqttMs      = 0;
uint32_t alertCooldownMs = 0;

// ─────────────────────────── Welford update ─────────────────────────────────
void welfordUpdate(WelfordState &s, double x) {
    s.count++;
    double delta  = x - s.mean;
    s.mean       += delta / s.count;
    double delta2 = x - s.mean;
    s.M2         += delta * delta2;
}

// ─────────────────────────── Ring buffer push ───────────────────────────────
void ringPush(float value) {
    ringBuf[ringHead] = value;
    ringHead = (ringHead + 1) % RING_BUFFER_SIZE;
    if (ringHead == 0) ringFull = true;
}

// Returns true if the sustained anomaly ratio exceeds ANOMALY_CONFIRM_RATIO.
bool ringAnomaly() {
    int  len     = ringFull ? RING_BUFFER_SIZE : ringHead;
    if (len == 0) return false;
    int  above   = 0;
    for (int i = 0; i < len; i++) {
        if (ringBuf[i] > 1.0f) above++;   // score > 1.0 means threshold breached
    }
    return ((float)above / len) >= ANOMALY_CONFIRM_RATIO;
}

// ─────────────────────────── Sensor reading ─────────────────────────────────
float readVibrationRMS() {
    sensors_event_t event;
    accel.getEvent(&event);
    // Euclidean magnitude of 3-axis acceleration
    float ax = event.acceleration.x;
    float ay = event.acceleration.y;
    float az = event.acceleration.z;
    return sqrt(ax*ax + ay*ay + az*az);
}

float readCurrentAmps() {
    return ina219.getCurrent_mA() / 1000.0f;   // convert mA → A
}

// ─────────────────────────── NVS persistence ────────────────────────────────
void saveBaseline() {
    prefs.begin("drumguard", false);
    prefs.putDouble("vib_mean",  vibState.mean);
    prefs.putDouble("vib_m2",    vibState.M2);
    prefs.putULong64("vib_cnt",  vibState.count);
    prefs.putDouble("cur_mean",  curState.mean);
    prefs.putDouble("cur_m2",    curState.M2);
    prefs.putULong64("cur_cnt",  curState.count);
    prefs.putDouble("aud_mean",  audioState.mean);
    prefs.putDouble("aud_m2",    audioState.M2);
    prefs.putULong64("aud_cnt",  audioState.count);
    prefs.putBool("calibrated",  true);
    prefs.end();
    Serial.println("[NVS] Baseline saved.");
}

bool loadBaseline() {
    prefs.begin("drumguard", true);
    bool ok = prefs.getBool("calibrated", false);
    if (ok) {
        vibState.mean  = prefs.getDouble("vib_mean", 0);
        vibState.M2    = prefs.getDouble("vib_m2",   0);
        vibState.count = prefs.getULong64("vib_cnt", 0);
        curState.mean  = prefs.getDouble("cur_mean", 0);
        curState.M2    = prefs.getDouble("cur_m2",   0);
        curState.count = prefs.getULong64("cur_cnt", 0);
        vibThreshold   = vibState.mean + SIGMA_MULTIPLIER * vibState.stddev();
        curThreshold   = curState.mean + SIGMA_MULTIPLIER * curState.stddev();
        audioState.mean  = prefs.getDouble("aud_mean", 0);
        audioState.M2    = prefs.getDouble("aud_m2",   0);
        audioState.count = prefs.getULong64("aud_cnt", 0);
        audioThreshold   = audioState.mean + SIGMA_MULTIPLIER * audioState.stddev();
    }
    prefs.end();
    return ok;
}

// ─────────────────────────── MQTT ───────────────────────────────────────────
void mqttReconnect() {
    while (!mqtt.connected()) {
        Serial.print("[MQTT] Connecting...");
        String clientId = String("DrumGuard-") + DEVICE_ID;
        if (mqtt.connect(clientId.c_str())) {
            Serial.println(" connected.");
            mqtt.publish("drumguard/status", "{\"status\":\"online\"}");
        } else {
            Serial.print(" failed (rc=");
            Serial.print(mqtt.state());
            Serial.println("). Retry in 5 s.");
            delay(5000);
        }
    }
}

void publishTelemetry(float vibRMS, float current, float anomalyScore,
                      const char* status) {
    StaticJsonDocument<256> doc;
    doc["device_id"]       = DEVICE_ID;
    doc["vibration_rms"]   = vibRMS;
    doc["current_amps"]    = current;
    doc["anomaly_score"]   = anomalyScore;
    doc["status"]          = status;
    doc["vib_threshold"]   = vibThreshold;
    doc["cur_threshold"]   = curThreshold;
    doc["audio_hf_energy"] = latestHighBandEnergy;
    doc["aud_threshold"]   = audioThreshold;
    doc["calibrated"]      = calibrated;
    
    char buf[256];
    serializeJson(doc, buf);
    mqtt.publish("drumguard/data", buf);
}

void publishAlert(float vibRMS, float current, float anomalyScore) {
    StaticJsonDocument<256> doc;
    doc["device_id"]       = DEVICE_ID;
    doc["vibration_rms"]   = vibRMS;
    doc["current_amps"]    = current;
    doc["anomaly_score"]   = anomalyScore;
    doc["status"]          = "WARNING";
    doc["message"]         = "Sustained anomaly detected. Fuser assembly may be degrading.";

    char buf[256];
    serializeJson(doc, buf);
    mqtt.publish("drumguard/alert", buf);
    Serial.println("[ALERT] Anomaly published!");
}

// ─────────────────────────── WiFi ───────────────────────────────────────────
void connectWiFi() {
    Serial.print("[WiFi] Connecting to ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("[WiFi] Connected. IP: ");
    Serial.println(WiFi.localIP());
}
// ────────────────────────── I2S ──────────────────────────────────────────────
void initI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 outputs 24-bit in 32-bit slot
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

//Core 0 RTOS 
void audioProcessingTask(void *pvParameters) {
    int32_t i2s_buffer[FFT_SAMPLES];
    size_t bytesIn;

    for (;;) {
        // Block until DMA fills the chunk
        i2s_read(I2S_PORT, &i2s_buffer, sizeof(i2s_buffer), &bytesIn, portMAX_DELAY);
        
        // Extract 24-bit payload from 32-bit slot and populate FFT
        for (int i = 0; i < FFT_SAMPLES; i++) {
            vReal[i] = (double)(i2s_buffer[i] >> 8); 
            vImag[i] = 0.0;
        }

        FFT.Windowing(FFT_WIN_TYP_HANNING, FFT_FORWARD);
        FFT.Compute(FFT_FORWARD);
        FFT.ComplexToMagnitude();

        // Integrate high-frequency bins
        double highBandEnergy = 0;
        double binWidth = (double)I2S_SAMPLE_RATE / FFT_SAMPLES; 
        int startBin = NOISE_FLOOR_HZ / binWidth; 
        
        for (int i = startBin; i < (FFT_SAMPLES / 2); i++) {
            highBandEnergy += vReal[i]; 
        }

        // Thread-safe write
        latestHighBandEnergy = (float)(highBandEnergy / 10000.0f); // Scale down for Welford sanity
        
        vTaskDelay(10 / portTICK_PERIOD_MS); // Yield
    }
}
// ─────────────────────────── Setup ──────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("=== DrumGuard v1.0 ===");

    // I2C
    Wire.begin(21, 22);   // SDA=GPIO21, SCL=GPIO22 (default ESP32)

    // ADXL345
    if (!accel.begin()) {
        Serial.println("[ERROR] ADXL345 not found. Check wiring.");
        while (1);
    }
    accel.setRange(ADXL345_RANGE_4_G);
    Serial.println("[OK] ADXL345 initialised.");

    // INA219
    if (!ina219.begin()) {
        Serial.println("[ERROR] INA219 not found. Check wiring.");
        while (1);
    }
    Serial.println("[OK] INA219 initialised.");

    // WiFi + MQTT
    connectWiFi();
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setKeepAlive(60);

    // Try to restore a saved baseline (survives reboots)
    if (loadBaseline()) {
        calibrated = true;
        Serial.printf("[NVS] Restored baseline: vib μ=%.4f σ=%.4f | cur μ=%.4f σ=%.4f\n",
                      vibState.mean, vibState.stddev(),
                      curState.mean, curState.stddev());
    } else {
        Serial.printf("[CAL] No baseline found. Starting %lu-second calibration window.\n",
                      CALIBRATION_DURATION_S);
        calibStartMs = millis();
    }

    initI2S();
    xTaskCreatePinnedToCore(
        audioProcessingTask,
        "Audio_DSP",
        10000,   // Stack size (FFT needs heap)
        NULL,
        1,       // Priority
        NULL,
        0        // Pin to Core 0
    );

    lastSampleMs = millis();
}

// ─────────────────────────── Loop ───────────────────────────────────────────
void loop() {
    // Keep MQTT alive
    if (!mqtt.connected()) mqttReconnect();
    mqtt.loop();

    uint32_t now = millis();
    if (now - lastSampleMs < SAMPLE_INTERVAL_MS) return;
    lastSampleMs = now;

    // Read sensors
    float vibRMS  = readVibrationRMS();
    float current = readCurrentAmps();
    float audioE  = latestHighBandEnergy;

    // ── Phase 1: Calibration ──────────────────────────────────────────────
    if (!calibrated) {
        welfordUpdate(vibState, vibRMS);
        welfordUpdate(curState, current);
        welfordUpdate(audioState, audioE);
        
        uint32_t elapsed = (now - calibStartMs) / 1000UL;
        if (elapsed >= CALIBRATION_DURATION_S) {
            vibThreshold = vibState.mean + SIGMA_MULTIPLIER * vibState.stddev();
            curThreshold = curState.mean + SIGMA_MULTIPLIER * curState.stddev();
            audioThreshold = audioState.mean + SIGMA_MULTIPLIER * audioState.stddev();
            calibrated   = true;
            saveBaseline();
            Serial.printf("[CAL] Calibration complete after %lu samples.\n", (unsigned long)vibState.count);
            Serial.printf("      Vib  threshold: %.4f g  (μ=%.4f σ=%.4f)\n",
                          vibThreshold, vibState.mean, vibState.stddev());
            Serial.printf("      Curr threshold: %.4f A  (μ=%.4f σ=%.4f)\n",
                          curThreshold, curState.mean, curState.stddev());
        }

        // Publish progress every 30 s during calibration
        if (now - lastMqttMs > 30000UL) {
            lastMqttMs = now;
            publishTelemetry(vibRMS, current, 0.0f, "calibrating");
        }
        return;
    }

    // ── Phase 2: Inference ────────────────────────────────────────────────
    // Normalise each sensor value: score = x / threshold
    // Score > 1.0 means the sample breached its individual threshold.
    float vibScore = (vibThreshold > 0) ? (float)(vibRMS  / vibThreshold) : 0.0f;
    float curScore = (curThreshold > 0) ? (float)(current / curThreshold) : 0.0f;
    float audScore = (audioThreshold > 0) ? (float)(audioE / audioThreshold) : 0.0f;
    // Combined anomaly score: max of the two normalised deviations.
    // Using max (not average) is conservative – either sensor breaching is enough.
    float anomalyScore = max(vibScore, max(curScore, audScore));

    ringPush(anomalyScore);

    bool anomaly = ringAnomaly();
    const char* status = anomaly ? "WARNING" : "OK";

    // Log to serial
    Serial.printf("[%s] Vib=%.3fg  Cur=%.3fA  Score=%.3f  RingFull=%d\n",
                  status, vibRMS, current, anomalyScore, (int)ringFull);

    // Publish telemetry every ~2 s
    if (now - lastMqttMs > 2000UL) {
        lastMqttMs = now;
        publishTelemetry(vibRMS, current, anomalyScore, status);
    }

    // Publish alert (with 60 s cooldown to avoid spam)
    if (anomaly && (now - alertCooldownMs > 60000UL)) {
        alertCooldownMs = now;
        publishAlert(vibRMS, current, anomalyScore);
    }
}
