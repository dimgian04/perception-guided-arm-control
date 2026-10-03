"""Standalone suction relay TCP server for the myArm Raspberry Pi.

Replaces the old ROS ``suction_relay_node``: instead of subscribing to a ROS
topic, it runs a plain TCP server (same JSON-line protocol as the joint TCP
server) so it can be driven over the network from the project workstation.

At all times it can:
  * report the current state of the pump and valve (state query), and
  * take commands to turn suction on/off, release (vacuum break), or set the
    pump/valve directly.

Wiring (BCM numbering):
    pump  -> GPIO 20
    valve -> GPIO 21

The relay module is ACTIVE-LOW, so a pin driven HIGH leaves its relay OFF and a
pin driven LOW switches it ON. Pins are initialised HIGH (everything OFF) and
returned HIGH on shutdown.

Protocol (one JSON object per line in, one JSON object per line out):
    {"type": "get_state"}                 -> current pump/valve state
    {"type": "on"}      / {"type":"grasp"}-> pump ON,  valve CLOSED  (suction)
    {"type": "off"}                       -> pump OFF, valve CLOSED  (idle)
    {"type": "release", "pulse_s": 0.3}   -> pump OFF, valve OPEN for a pulse,
                                             then CLOSED (breaks the vacuum)
    {"type": "set", "pump": true/false, "valve": true/false}

Every response echoes the resulting state, e.g.:
    {"ok": true, "action": "on", "pump": "on", "valve": "closed",
     "pump_pin": 20, "valve_pin": 21, "pump_level": "LOW",
     "valve_level": "HIGH", "dry_run": false}

Run on the Pi:
    python3 suction_tcp_server.py
    python3 suction_tcp_server.py --pump-pin 20 --valve-pin 21 --port 5018
"""

import argparse
import json
import socket
import threading
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


DEFAULT_PUMP_PIN = 20
DEFAULT_VALVE_PIN = 21
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_BIND_PORT = 5018
DEFAULT_RELEASE_PULSE_S = 0.3


def log(message):
    print(f"[suction-tcp] {time.strftime('%H:%M:%S')} {message}", flush=True)


class SuctionRelayServer:
    def __init__(self, pump_pin, valve_pin, host, port, release_pulse_s, dry_run):
        self.pump_pin = int(pump_pin)
        self.valve_pin = int(valve_pin)
        self.host = host
        self.port = int(port)
        self.release_pulse_s = float(release_pulse_s)
        self.dry_run = bool(dry_run) or GPIO is None

        self.lock = threading.Lock()
        self.pump_active = False
        self.valve_open = False
        self.release_timer = None
        self.shutdown_event = threading.Event()
        self.server_socket = None

        if self.dry_run:
            log("Running in DRY-RUN mode (no GPIO access).")
        else:
            GPIO.setmode(GPIO.BCM)
            # Active-low relay: initialise HIGH so both relays start OFF.
            GPIO.setup(self.pump_pin, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(self.valve_pin, GPIO.OUT, initial=GPIO.HIGH)

        # Force a known, safe state at startup: pump OFF, valve CLOSED.
        self.apply(pump_active=False, valve_open=False)
        log(f"Suction relay ready. pump_pin={self.pump_pin} valve_pin={self.valve_pin} "
            f"(active-low, default HIGH=off)")

    # ------------------------------------------------------------
    # GPIO state.
    # ------------------------------------------------------------

    def apply(self, pump_active, valve_open):
        with self.lock:
            self.pump_active = bool(pump_active)
            self.valve_open = bool(valve_open)
            if not self.dry_run:
                # Active-low: LOW = relay ON, HIGH = relay OFF.
                GPIO.output(self.pump_pin, GPIO.LOW if self.pump_active else GPIO.HIGH)
                GPIO.output(self.valve_pin, GPIO.LOW if self.valve_open else GPIO.HIGH)

    def state(self):
        with self.lock:
            return {
                "ok": True,
                "pump": "on" if self.pump_active else "off",
                "valve": "open" if self.valve_open else "closed",
                "pump_pin": self.pump_pin,
                "valve_pin": self.valve_pin,
                "pump_level": "LOW" if self.pump_active else "HIGH",
                "valve_level": "LOW" if self.valve_open else "HIGH",
                "dry_run": self.dry_run,
            }

    def respond(self, action):
        result = self.state()
        result["action"] = action
        return result

    # ------------------------------------------------------------
    # Actions.
    # ------------------------------------------------------------

    def cancel_release_timer(self):
        if self.release_timer is not None:
            self.release_timer.cancel()
            self.release_timer = None

    def suction_on(self):
        self.cancel_release_timer()
        self.apply(pump_active=True, valve_open=False)
        log("Suction ON: pump ON, valve CLOSED")

    def suction_off(self):
        self.cancel_release_timer()
        self.apply(pump_active=False, valve_open=False)
        log("Suction OFF: pump OFF, valve CLOSED")

    def release(self, pulse_s):
        self.cancel_release_timer()
        self.apply(pump_active=False, valve_open=True)
        self.release_timer = threading.Timer(pulse_s, self.finish_release)
        self.release_timer.daemon = True
        self.release_timer.start()
        log(f"Release: pump OFF, valve OPEN for {pulse_s:.2f} s")

    def finish_release(self):
        self.apply(pump_active=False, valve_open=False)
        with self.lock:
            self.release_timer = None
        log("Release pulse complete: valve CLOSED")

    # ------------------------------------------------------------
    # Command handling.
    # ------------------------------------------------------------

    def handle_command(self, command):
        command_type = str(command.get("type", "")).lower()

        if command_type in ("get_state", "status", "state", "get_status"):
            return self.respond("get_state")

        if command_type in ("on", "grasp", "suction_on", "pump_on"):
            self.suction_on()
            return self.respond("on")

        if command_type in ("off", "suction_off", "stop", "idle"):
            self.suction_off()
            return self.respond("off")

        if command_type in ("release", "blow", "drop"):
            pulse_s = float(command.get("pulse_s", self.release_pulse_s))
            self.release(pulse_s)
            return self.respond("release")

        if command_type == "set":
            pump = bool(command.get("pump", self.pump_active))
            valve = bool(command.get("valve", self.valve_open))
            self.cancel_release_timer()
            self.apply(pump_active=pump, valve_open=valve)
            log(f"Set: pump {'ON' if pump else 'OFF'}, valve {'OPEN' if valve else 'CLOSED'}")
            return self.respond("set")

        return {"ok": False, "error": f"unknown command type: {command.get('type')!r}"}

    # ------------------------------------------------------------
    # TCP server.
    # ------------------------------------------------------------

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self.server_socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            server.settimeout(0.5)
            log(f"Suction TCP server listening on {self.host}:{self.port}")

            while not self.shutdown_event.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with client:
                    client.settimeout(2.0)
                    self.handle_client(client, address)

    def handle_client(self, client, address):
        try:
            line = self.read_line(client)
            command = json.loads(line)
            result = self.handle_command(command)
        except Exception as exc:  # noqa: BLE001 - report any client/parse error back.
            result = {"ok": False, "error": str(exc)}
        try:
            client.sendall((json.dumps(result) + "\n").encode("utf-8"))
        except OSError:
            pass

    @staticmethod
    def read_line(sock):
        chunks = []
        while True:
            chunk = sock.recv(1)
            if not chunk:
                break
            if chunk == b"\n":
                break
            chunks.append(chunk)
        if not chunks:
            raise ValueError("Empty TCP command")
        return b"".join(chunks).decode("utf-8")

    def shutdown(self):
        self.shutdown_event.set()
        self.cancel_release_timer()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        # Leave the rig safe: everything OFF (pins HIGH) and release the GPIO.
        self.apply(pump_active=False, valve_open=False)
        if not self.dry_run:
            GPIO.cleanup([self.pump_pin, self.valve_pin])
        log("Suction relay shut down (pump OFF, valve CLOSED, GPIO released).")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone suction relay TCP server.")
    parser.add_argument("--pump-pin", type=int, default=DEFAULT_PUMP_PIN, help="BCM pin for the pump relay")
    parser.add_argument("--valve-pin", type=int, default=DEFAULT_VALVE_PIN, help="BCM pin for the valve relay")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_BIND_PORT, help="Bind port")
    parser.add_argument("--release-pulse-s", type=float, default=DEFAULT_RELEASE_PULSE_S,
                        help="How long the valve stays open on a release command")
    parser.add_argument("--dry-run", action="store_true", help="Do not touch GPIO (for testing off-Pi)")
    return parser.parse_args()


def main():
    args = parse_args()
    server = SuctionRelayServer(
        pump_pin=args.pump_pin,
        valve_pin=args.valve_pin,
        host=args.host,
        port=args.port,
        release_pulse_s=args.release_pulse_s,
        dry_run=args.dry_run,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
