"""
drumguard_monitor.py
─────────────────────────────────────────────────────────────────────────────
DrumGuard – Real-time MQTT subscriber and terminal dashboard.

Subscribes to drumguard/# and pretty-prints telemetry and alerts.

Usage:
  pip install paho-mqtt rich
  python drumguard_monitor.py [--broker broker.hivemq.com] [--port 1883]
"""

import argparse
import json
import sys
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit("Install paho-mqtt:  pip install paho-mqtt")

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.text    import Text
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    print("[INFO] Install 'rich' for a nicer display:  pip install rich")


TOPICS = [
    "drumguard/data",
    "drumguard/status",
    "drumguard/alert",
]


def format_score(score: float) -> str:
    if score > 1.2:  return f"[bold red]  {score:.3f}  !!ALERT!!"
    if score > 1.0:  return f"[yellow]{score:.3f} (marginal)"
    return f"[green]{score:.3f} (OK)"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        if RICH:
            console.print(f"[bold green]✓ Connected to MQTT broker[/bold green]")
        else:
            print("✓ Connected to MQTT broker")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"  Subscribed: {topic}")
    else:
        print(f"[ERROR] Connection failed (rc={rc})")


def on_message(client, userdata, msg):
    local_ts = datetime.now().strftime("%H:%M:%S")
    raw = msg.payload.decode("utf-8", errors="replace")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[{local_ts}] {msg.topic}: {raw}")
        return

    # Prefer the ISO timestamp from the payload; fall back to local clock
    ts = payload.get("timestamp", local_ts)
    topic = msg.topic

    if topic == "drumguard/alert":
        if RICH:
            console.rule("[bold red]⚠  ANOMALY ALERT  ⚠")
            console.print(
                f"  Time    : [bold]{ts}[/bold]\n"
                f"  Device  : [bold]{payload.get('device_id','?')}[/bold]\n"
                f"  Vib RMS : {payload.get('vibration_rms','?')} g\n"
                f"  Current : {payload.get('current_amps','?')} A\n"
                f"  Score   : [bold red]{payload.get('anomaly_score','?')}[/bold red]\n"
                f"  Message : {payload.get('message','')}"
            )
            console.rule()
        else:
            print(f"\n{'='*60}")
            print(f"⚠  ALERT  [{ts}]  {payload}")
            print('='*60)

    elif topic == "drumguard/data":
        status = payload.get("status", "?")
        score  = float(payload.get("anomaly_score", 0))
        vib    = payload.get("vibration_rms", "?")
        cur    = payload.get("current_amps", "?")
        cal    = payload.get("calibrated", False)

        if RICH:
            color  = "red" if score > 1.2 else ("yellow" if score > 1.0 else "green")
            status_text = Text(f"[{status}]", style=f"bold {color}")
            console.print(
                f"[dim]{ts}[/dim] "
                f"{'[dim]CAL[/dim]' if not cal else ''}"
                f"Vib=[cyan]{vib}[/cyan]g  "
                f"Cur=[cyan]{cur}[/cyan]A  "
                f"Score={format_score(score)}"
            )
        else:
            print(f"[{ts}] {status:<12} Vib={vib}g  Cur={cur}A  Score={score:.3f}")

    elif topic == "drumguard/status":
        if RICH:
            console.print(f"[dim]{ts}[/dim] [bold blue]STATUS[/bold blue] {raw}")
        else:
            print(f"[{ts}] STATUS: {raw}")


def main():
    parser = argparse.ArgumentParser(description="DrumGuard MQTT monitor")
    parser.add_argument("--broker", default="broker.hivemq.com")
    parser.add_argument("--port",   type=int, default=1883)
    args = parser.parse_args()

    client = mqtt.Client(client_id="drumguard-monitor")
    client.on_connect = on_connect
    client.on_message = on_message

    if RICH:
        console.print(f"[bold]DrumGuard Monitor[/bold] → {args.broker}:{args.port}")
    else:
        print(f"DrumGuard Monitor → {args.broker}:{args.port}")

    try:
        client.connect(args.broker, args.port, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
