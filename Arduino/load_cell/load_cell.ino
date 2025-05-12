#include <HX711_ADC.h>
#if defined(ESP8266)|| defined(ESP32) || defined(AVR)
#include <EEPROM.h>
#endif

// Pin definitions for the load cells
const int LOADCELL_DOUT_PIN_X = 2; // X-axis load cell data pin
const int LOADCELL_SCK_PIN_X = 3;  // X-axis load cell clock pin

//HX711 constructor:
HX711_ADC LoadCellX(LOADCELL_DOUT_PIN_X, LOADCELL_SCK_PIN_X);

const int calVal_eepromAdress = 0;
unsigned long t = 0;

void setup() {
  Serial.begin(57600); delay(10);
  //Serial.println();
  //Serial.println("Starting...");
  LoadCellX.begin();
  //LoadCell.setReverseOutput(); //uncomment to turn a negative output value to positive
  unsigned long stabilizingtime = 2000; // preciscion right after power-up can be improved by adding a few seconds of stabilizing time
  boolean _tare = true; //set this to false if you don't want tare to be performed in the next step
  LoadCellX.start(stabilizingtime, _tare);
  if (LoadCellX.getTareTimeoutFlag() || LoadCellX.getSignalTimeoutFlag()) {
    Serial.println("Timeout, check MCU>HX711 wiring and pin designations");
    while (1);
  }
  else {
    LoadCellX.setCalFactor(107500.0); // user set calibration value (float), initial value 1.0 may be used for this sketch
  }
  while (!LoadCellX.update());
  calibrate(); //start calibration procedure
  loop();
}

void loop() {
  static boolean newDataReady = 0;
  const int serialPrintInterval = 0; //increase value to slow down serial print activity

  // check for new data/start next conversion:
  if (LoadCellX.update()) newDataReady = true;

  if (newDataReady) {
      float loadX = -1.00*LoadCellX.getData();
      Serial.println(loadX, 3);
      newDataReady = 0;
      delay(100);
  }


  // receive command from serial terminal
  if (Serial.available() > 0) {
    char inByte = Serial.read();
    if (inByte == 't') {
      LoadCellX.tareNoDelay(); //tare
    }
    else if (inByte == 'r') re_calibrate(); //calibrate
  }

}

void calibrate() {
  Serial.println("Send 't' from serial monitor to set the tare offset.");

  bool _resume = false;
  bool tare = false;
  while (_resume == false) {
    LoadCellX.update();
    if (tare == false){
      LoadCellX.tareNoDelay();
      tare = true;
      _resume = true;
    }
  }
}

void re_calibrate() {
  Serial.println("Place your known mass on the loadcell.");
  Serial.println("Then send the weight of this mass (i.e. 100.0) from serial monitor.");

  float known_mass = 0;
  boolean _resume = false;;
  while (_resume == false) {
    LoadCellX.update();
    if (Serial.available() > 0) {
      known_mass = Serial.parseFloat();
      if (known_mass != 0) {
        Serial.print("Known mass is: ");
        Serial.println(known_mass);
        _resume = true;
      }
    }
  }
  LoadCellX.refreshDataSet(); //refresh the dataset to be sure that the known mass is measured correct
  float newCalibrationValueX = LoadCellX.getNewCalibration(known_mass); //get the new calibration value
  LoadCellX.setCalFactor(newCalibrationValueX);
  Serial.print("New calibration value has been set to: ");
  Serial.print(newCalibrationValueX);
  Serial.print(" (X-axis)");
  Serial.println("Use these calibration values (calFactor) in your project sketch.");
  delay(5000);
}


  

 
