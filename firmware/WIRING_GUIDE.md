# Hardware Wiring & Flashing Guide

## Components Required

| Item | Qty | Notes |
|---|---|---|
| ESP32 DevKit V1 | 1 | Microcontroller with micro-USB cable |
| FSR Pressure Sensors | 3 | Heel, Midfoot, Forefoot |
| 10 kΩ Resistors | 3 | Pull-down resistors for FSR voltage dividers |
| MPU6050 6-DOF IMU Module | 1 | Accelerometer + Gyroscope |
| Breadboard & Jumper Wires | 1 | Standard prototyping set |

---

## Circuit Schematic

### 1. FSR Pressure Sensors (Voltage Divider)
Each FSR changes its resistance when pressure is applied. To measure this as a voltage on the ESP32 analog pins:
- Connect **one leg of each FSR** to the ESP32 **3.3V** pin.
- Connect the **other leg of the FSR** to both:
  1. An ESP32 **GPIO Analog Pin**.
  2. A **10 kΩ resistor** connected to **GND**.

```
3.3V ─────── [ FSR ] ───────┬───────> ESP32 GPIO Pin (Analog)
                            │
                        [10k Resistor]
                            │
                           GND
```

#### Pin Assignment:
- **Heel FSR:** GPIO 34 (ADC1_CH6)
- **Midfoot FSR:** GPIO 35 (ADC1_CH7)
- **Forefoot FSR:** GPIO 32 (ADC1_CH4)

---

### 2. MPU6050 Motion Sensor (I2C)
The MPU6050 communicates with the ESP32 using the I2C protocol:
- **VCC** ──> ESP32 **3.3V** (or 5V if your module has an onboard regulator)
- **GND** ──> ESP32 **GND**
- **SDA** ──> ESP32 **GPIO 21**
- **SCL** ──> ESP32 **GPIO 22**
- **AD0** ──> ESP32 **GND** (sets I2C address to `0x68`)

---

## Flashing the ESP32 with Arduino IDE

1. **Install Arduino IDE** from [arduino.cc](https://www.arduino.cc/en/software).
2. Go to **Settings / Preferences** and add the ESP32 board URL:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Open **Boards Manager**, search for `esp32`, and install **esp32 by Espressif Systems**.
4. Open the firmware file:
   `firmware/esp32_gait_acquisition/esp32_gait_acquisition.ino`
5. Connect your ESP32 to your laptop using a micro-USB cable.
6. Select **Tools → Board → ESP32 Dev Module** and choose your serial port under **Tools → Port**.
7. Click **Upload** (➡️).
8. Once uploaded, open **Tools → Serial Monitor** set to **115200 baud** to verify that JSON messages stream at 50 Hz:
   ```json
   {"timestamp_ms":1250,"heel":120,"midfoot":240,"forefoot":180,"acc_x_g":0.12,"acc_y_g":0.25,"acc_z_g":0.98,"gyro_x_dps":0.3,"gyro_y_dps":0.1,"gyro_z_dps":0.4}
   ```
9. Close the Serial Monitor before launching the Python dashboard, as only one program can access the USB serial port at a time.

