//Libraries
#include <DHT.h>;
#include <SD.h>
#include <SPI.h>
File myFile;
char fileName[] = "sensor_reading.csv";
const int chipSelect = 10;

//Constants
#define DHTPIN 3     // what pin we're connected to
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
  if (SD.begin(chipSelect))
  {
    Serial.println("SD card is present & ready");
    deleteFile();
    writeToFile("SecPassed,Temp,Humidity");
  } 
  else
  {
    Serial.println("SD card missing or failure");
    while(1);  //wait here forever
  }
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
    writeToFile(buf.c_str());
    sec = sec + 5;
    delay(5000); 
}

void writeToFile(char* content)
{
  myFile = SD.open(fileName, FILE_WRITE);
  if (myFile) // it opened OK
    {
      Serial.println("Writing to a file...");
      myFile.println(content);
      myFile.close(); 
      Serial.println("Done");
    }
  else 
    Serial.println("Error opening simple.txt");
}

void deleteFile()
{
 //delete a file:
  if (SD.exists(fileName)) 
  {
    Serial.println("Removing file..");
    SD.remove(fileName);
    Serial.println("Done");
  } 
}

   
