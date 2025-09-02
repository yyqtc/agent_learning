def get_web_code(url: str) -> str:
   """获取网页代码"""
   import requests

   headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
      'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1',
   }

   response = requests.get(url, headers=headers)
   return response.text

def revert_pound_to_kg(pound: float) -> float:
   """将磅转换为千克"""
   return pound * 0.453592

def revert_meter_to_cm(meter: float) -> float:
   """将厘米转换为米"""
   return meter / 100

def compute_bmi(weight: float, height: float) -> float:
   """计算BMI"""
   return weight / (height ** 2)

def get_city_weather(city: str) -> str:
   """获取城市天气"""
   import json
   import requests
   
   config = json.load(open("config.json", "r"))
   api_key = config["QWeather-API-KEY"]
   api_base = config["QWeather-API-BASE"]
   
   location_url = f"{api_base}/geo/v2/city/lookup?location={city}"

   weather_url = f"{api_base}/v7/weather/3d"
   response = requests.get(url, headers={"Authorization": f'Bearer {api_key}'})
   return response.json()
   