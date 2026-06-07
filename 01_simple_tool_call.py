import json
import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Weather(str, Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    CLOUDY = "cloudy"
    SNOWY = "snowy"


WEATHER_DATA = {
    "vancouver": Weather.RAINY,
    "dubai": Weather.SUNNY,
    "tokyo": Weather.CLOUDY,
}

ATTRACTIONS = {
    "vancouver": [
        {"name": "Stanley Park", "best_weather": Weather.SUNNY, "note": "Great for walking and cycling"},
        {"name": "Granville Island Market", "best_weather": Weather.RAINY, "note": "Perfect indoor market for rainy days"},
        {"name": "Capilano Suspension Bridge", "best_weather": Weather.CLOUDY, "note": "Scenic forest experience"},
        {"name": "Grouse Mountain", "best_weather": Weather.SNOWY, "note": "Skiing and snowshoeing in winter"},
        {"name": "Gastown Steam Clock", "best_weather": Weather.CLOUDY, "note": "Historic district, nice any day"},
    ],
    "dubai": [
        {"name": "Burj Khalifa", "best_weather": Weather.SUNNY, "note": "Best views on clear sunny days"},
        {"name": "Dubai Mall", "best_weather": Weather.RAINY, "note": "Massive indoor shopping and entertainment"},
        {"name": "Desert Safari", "best_weather": Weather.SUNNY, "note": "Dune bashing and camel rides"},
        {"name": "Dubai Creek", "best_weather": Weather.CLOUDY, "note": "Historic waterway, pleasant when not too hot"},
        {"name": "Palm Jumeirah", "best_weather": Weather.SUNNY, "note": "Iconic island best enjoyed in sunshine"},
    ],
    "tokyo": [
        {"name": "Senso-ji Temple", "best_weather": Weather.CLOUDY, "note": "Ancient temple, atmospheric any weather"},
        {"name": "Shinjuku Gyoen Garden", "best_weather": Weather.SUNNY, "note": "Beautiful gardens for picnics"},
        {"name": "teamLab Borderless", "best_weather": Weather.RAINY, "note": "Immersive indoor digital art museum"},
        {"name": "Mount Fuji Day Trip", "best_weather": Weather.SUNNY, "note": "Best views on clear days"},
        {"name": "Tsukiji Outer Market", "best_weather": Weather.CLOUDY, "note": "Fresh seafood, great morning visit"},
    ],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (lowercase)",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction",
            "description": "Get tourist attractions for a city given the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (lowercase)",
                    },
                    "weather": {
                        "type": "string",
                        "enum": [w.value for w in Weather],
                        "description": "Current weather condition",
                    },
                },
                "required": ["city", "weather"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful Travel Assistant. "
    "When asked about a city, first check the current weather using get_weather, "
    "then recommend attractions suited to that weather using get_attraction. "
    "Present the recommendations in a friendly, concise way."
)


def get_weather(city: str) -> str:
    city = city.lower()
    if city not in WEATHER_DATA:
        return f"Weather data not available for {city}."
    weather = WEATHER_DATA[city]
    return f"The current weather in {city.title()} is {weather.value}."


def get_attraction(city: str, weather: Weather) -> str:
    city = city.lower()
    if city not in ATTRACTIONS:
        return f"No attraction data available for {city}."
    if isinstance(weather, str):
        weather = Weather(weather)
    attractions = ATTRACTIONS[city]
    lines = [f"Top attractions in {city.title()} (weather: {weather.value}):"]
    for i, attr in enumerate(attractions, 1):
        tag = " [RECOMMENDED for today]" if attr["best_weather"] == weather else ""
        lines.append(f"  {i}. {attr['name']}{tag} — {attr['note']}")
    return "\n".join(lines)


def dispatch_tool(name: str, arguments: dict) -> str:
    if name == "get_weather":
        return get_weather(**arguments)
    elif name == "get_attraction":
        return get_attraction(**arguments)
    else:
        return f"Unknown tool: {name}"


class TravelAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_URL"),
        )
        self.model = os.getenv("LLM_MODEL")
        print(f"[INIT] Model: {self.model}")
        print(f"[INIT] Base URL: {os.getenv('LLM_URL')}")

    def run(self, city: str):
        user_prompt = f"I'm planning to visit {city}. What should I do there today?"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print(f"\n{'='*60}")
        print(f"[USER] {user_prompt}")
        print(f"{'='*60}")

        step = 0
        while True:
            step += 1
            print(f"\n[STEP {step}] Calling LLM...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            print(f"[STEP {step}] finish_reason={finish_reason}")

            if message.content:
                print(f"[STEP {step}] LLM text: {message.content}")

            if finish_reason == "stop" or not message.tool_calls:
                print(f"\n{'='*60}")
                print("[FINAL ANSWER]")
                print(message.content)
                print(f"{'='*60}")
                break

            messages.append(message)

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"[TOOL CALL] {name}({arguments})")

                result = dispatch_tool(name, arguments)
                print(f"[TOOL RESULT] {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })


if __name__ == "__main__":
    assistant = TravelAssistant()
    assistant.run(city="vancouver")



"""
[INIT] Model: gemma-4-31b-it
[INIT] Base URL: https://generativelanguage.googleapis.com/v1beta/openai/

============================================================
[USER] I'm planning to visit vancouver. What should I do there today?
============================================================

[STEP 1] Calling LLM...
[STEP 1] finish_reason=tool_calls
[STEP 1] LLM text: <thought>The user wants recommendations for things to do in Vancouver today.
According to the system instructions, I should:
1.  Call `get_weather` for "vancouver".
2.  Based on the weather result, call `get_attraction` for "vancouver" and the specific weather condition.
3.  Present the recommendations to the user.

Step 1: Call `get_weather(city='vancouver')`.</thought>
[TOOL CALL] get_weather({'city': 'vancouver'})
[TOOL RESULT] The current weather in Vancouver is rainy.

[STEP 2] Calling LLM...
[STEP 2] finish_reason=tool_calls
[STEP 2] LLM text: <thought><thought>The weather in Vancouver is rainy. Now I need to get attractions suited for rainy weather.
Step 2: Call `get_attraction(city='vancouver', weather='rainy')`.</thought></thought>
[TOOL CALL] get_attraction({'city': 'vancouver', 'weather': 'rainy'})
[TOOL RESULT] Top attractions in Vancouver (weather: rainy):
  1. Stanley Park — Great for walking and cycling
  2. Granville Island Market [RECOMMENDED for today] — Perfect indoor market for rainy days
  3. Capilano Suspension Bridge — Scenic forest experience
  4. Grouse Mountain — Skiing and snowshoeing in winter
  5. Gastown Steam Clock — Historic district, nice any day

[STEP 3] Calling LLM...
[STEP 3] finish_reason=stop
[STEP 3] LLM text: <thought><thought>The weather is rainy and the recommended attraction is the Granville Island Market. I should present this to the user in a friendly and concise way.</thought></thought></thought>It's currently rainy in Vancouver, so it's a great day to stay dry! I highly recommend visiting the **Granville Island Market**—it's a perfect indoor spot to explore local crafts and delicious food while avoiding the rain. Enjoy your visit!

============================================================
[FINAL ANSWER]
<thought><thought>The weather is rainy and the recommended attraction is the Granville Island Market. I should present this to the user in a friendly and concise way.</thought></thought></thought>It's currently rainy in Vancouver, so it's a great day to stay dry! I highly recommend visiting the **Granville Island Market**—it's a perfect indoor spot to explore local crafts and delicious food while avoiding the rain. Enjoy your visit!
============================================================

"""