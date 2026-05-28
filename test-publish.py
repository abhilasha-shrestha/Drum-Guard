import paho.mqtt.publish as publish
import json

payload = {
    "device_id": "printer_01",
    "vibration_rms": 1.45,
    "current_amps": 1.88,
    "anomaly_score": 1.32,
    "status": "WARNING"
}

publish.single(
    "drumguard/alert",
    json.dumps(payload),
    hostname="broker.hivemq.com"
)

print("Message sent")