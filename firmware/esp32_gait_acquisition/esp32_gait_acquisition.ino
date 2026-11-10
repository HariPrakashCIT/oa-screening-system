/*
  ==============================================================================
  AI-Based Osteoarthritis Screening System — ESP32 Firmware
  ==============================================================================
  Microcontroller : ESP32 DevKit V1 (30 or 36 pins)
  Sensors         : 3x FSR Pressure Sensors + 1x MPU6050 6-DOF IMU
  Communication   : USB Serial @ 115200 baud
  Protocol        : Single line JSON at ~50 Hz (every 20 ms)
  
  HARDWARE WIRING:
  ----------------
  1. FSR Sensors (Voltage Divider Circuit):
     - Each FSR connects between 3.3V and an Analog Pin.
     - A 10k Ohm pull-down resistor connects between that Analog Pin and GND.
     - FSR 1 (Heel)     ---> ESP32 GPIO 34 (ADC1_CH6)
     - FSR 2 (Midfoot)  ---> ESP32 GPIO 35 (ADC1_CH7)
     - FSR 3 (Forefoot) ---> ESP32 GPIO 32 (ADC1_CH4)
  
  2. MPU6050 IMU (I2C):
     - VCC ---> ESP32 3.3V (or 5V if module has onboard 3.3V regulator)
     - GND ---> ESP32 GND
     - SDA ---> ESP32 GPIO 21
     - SCL ---> ESP32 GPIO 22
  ==============================================================================
*/

#include <Wire.h>

// -----------------------------------------------------------------------------
// Pin Definitions
// -----------------------------------------------------------------------------
const int PIN_FSR_HEEL     = 34; // Analog input for heel FSR
const int PIN_FSR_MIDFOOT  = 35; // Analog input for midfoot FSR
const int PIN_FSR_FOREFOOT = 32; // Analog input for forefoot FSR

// -----------------------------------------------------------------------------
// MPU6050 I2C Configuration
// -----------------------------------------------------------------------------
const int MPU6050_ADDR = 0x68; // Default I2C address (AD0 to GND)

// Sensor scaling factors for default sensitivity ranges:
// Accelerometer: +/- 2g range -> 16384 LSB/g
const float ACCEL_SCALE = 16384.0;
// Gyroscope: +/- 250 deg/s range -> 131.0 LSB/(deg/s)
const float GYRO_SCALE  = 131.0;

// Sampling interval for ~50 Hz rate (20 milliseconds)
const unsigned long SAMPLE_INTERVAL_MS = 20;
unsigned long lastSampleTime = 0;

// Calibration offsets (calculated at startup)
float ax_offset = 0.0, ay_offset = 0.0, az_offset = 0.0;
float gx_offset = 0.0, gy_offset = 0.0, gz_offset = 0.0;

// -----------------------------------------------------------------------------
// Function Prototypes
// -----------------------------------------------------------------------------
void initMPU6050();
void readMPU6050(float &ax, float &ay, float &az, float &gx, float &gy, float &gz);

void setup() {
  // Initialize USB serial communication
  Serial.begin(115200);
  while (!Serial && millis() < 2000); // Wait for serial connection

  // Configure ADC attenuation for full 0 - 3.3V range (12-bit: 0 - 4095)
  analogSetAttenuation(ADC_11db);
  pinMode(PIN_FSR_HEEL, INPUT);
  pinMode(PIN_FSR_MIDFOOT, INPUT);
  pinMode(PIN_FSR_FOREFOOT, INPUT);

  // Initialize I2C communication
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22
  Wire.setClock(400000); // 400 kHz fast mode I2C

  // Initialize MPU6050
  initMPU6050();

  // Print startup ready comment
  Serial.println("// ESP32 OA Screening Firmware Initialized at 50 Hz.");
}

void loop() {
  unsigned long currentMillis = millis();

  // Enforce 50 Hz sampling cadence (every 20 ms)
  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentMillis;

    // 1. Read Analog Pressure Values (12-bit: 0 to 4095)
    int heel     = analogRead(PIN_FSR_HEEL);
    int midfoot  = analogRead(PIN_FSR_MIDFOOT);
    int forefoot = analogRead(PIN_FSR_FOREFOOT);

    // 2. Read IMU Kinematics
    float ax, ay, az, gx, gy, gz;
    readMPU6050(ax, ay, az, gx, gy, gz);

    // 3. Emit Single-Line Valid JSON Payload
    // Expected format:
    // {"timestamp_ms":1250,"heel":120,"midfoot":240,"forefoot":180,"acc_x_g":0.12,"acc_y_g":0.25,"acc_z_g":0.98,"gyro_x_dps":0.3,"gyro_y_dps":0.1,"gyro_z_dps":0.4}
    Serial.print("{\"timestamp_ms\":");
    Serial.print(currentMillis);
    Serial.print(",\"heel\":");
    Serial.print(heel);
    Serial.print(",\"midfoot\":");
    Serial.print(midfoot);
    Serial.print(",\"forefoot\":");
    Serial.print(forefoot);
    Serial.print(",\"acc_x_g\":");
    Serial.print(ax, 3);
    Serial.print(",\"acc_y_g\":");
    Serial.print(ay, 3);
    Serial.print(",\"acc_z_g\":");
    Serial.print(az, 3);
    Serial.print(",\"gyro_x_dps\":");
    Serial.print(gx, 2);
    Serial.print(",\"gyro_y_dps\":");
    Serial.print(gy, 2);
    Serial.print(",\"gyro_z_dps\":");
    Serial.print(gz, 2);
    Serial.println("}");
  }
}

// -----------------------------------------------------------------------------
// Initialize MPU6050 Wakeup
// -----------------------------------------------------------------------------
void initMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0x00); // Wake up sensor from sleep mode
  byte err = Wire.endTransmission();

  if (err != 0) {
    Serial.println("// WARNING: MPU6050 not detected on I2C bus (0x68). Check wiring.");
  }
}

// -----------------------------------------------------------------------------
// Read Raw IMU Registers & Convert to Units (g and deg/s)
// -----------------------------------------------------------------------------
void readMPU6050(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B); // Starting register for accelerometer data (ACCEL_XOUT_H)
  byte err = Wire.endTransmission(false);

  if (err != 0) {
    // If I2C read fails, return neutral gravity along Z
    ax = 0.0; ay = 0.0; az = 1.0;
    gx = 0.0; gy = 0.0; gz = 0.0;
    return;
  }

  // Request 14 sequential bytes:
  // Accel X (2), Accel Y (2), Accel Z (2), Temp (2), Gyro X (2), Gyro Y (2), Gyro Z (2)
  Wire.requestFrom(MPU6050_ADDR, 14, true);

  if (Wire.available() >= 14) {
    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    
    // Skip temperature bytes
    Wire.read(); Wire.read();
    
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    // Convert raw signed integer counts into physical units
    ax = ((float)raw_ax / ACCEL_SCALE);
    ay = ((float)raw_ay / ACCEL_SCALE);
    az = ((float)raw_az / ACCEL_SCALE);

    gx = ((float)raw_gx / GYRO_SCALE);
    gy = ((float)raw_gy / GYRO_SCALE);
    gz = ((float)raw_gz / GYRO_SCALE);
  } else {
    ax = 0.0; ay = 0.0; az = 1.0;
    gx = 0.0; gy = 0.0; gz = 0.0;
  }
}

