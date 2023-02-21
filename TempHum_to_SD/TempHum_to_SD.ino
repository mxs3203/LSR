//Libraries
#include <DHT.h>;
#include <SD.h>
#include <SPI.h>
File myFile;
const int chipSelect = 4;

//Constants
#define DHTPIN 2     // what pin we're connected to
#define DHTTYPE DHT22   // DHT 22  (AM2302)
DHT dht(DHTPIN, DHTTYPE); //// Initialize DHT sensor for normal 16mhz Arduino


//Variables
float hum;  //Stores humidity value
float temp; //Stores temperature value
int sec = 0;

void setup()
{
  Serial.begin(9600);
  dht.begin();
  if (!SD.begin(chipSelect))
  {
    Serial.println("SD card missing or failure");
    while(1);  //wait here forever
  }
  SD.remove("datalog.txt");
}

void loop()
{
    hum = dht.readHumidity();
    temp = dht.readTemperature();
    String buf  = String(sec) + "," + String(temp) + "," + String(hum);
    Serial.print("Humidity: ");
    Serial.print(hum);
    Serial.print(" %, Temp: ");
    Serial.print(temp);
    Serial.println(" Celsius");
    File dataFile = SD.open("datalog.txt", FILE_WRITE);

  // if the file is available, write to it:
  if (dataFile) {
    dataFile.println(buf);
    dataFile.close();
    // print to the serial port too:
    Serial.println(buf);
  }
  else {
    Serial.println("error opening datalog.txt");
  }
  sec = sec + 30;
  delay(30000); 
}
