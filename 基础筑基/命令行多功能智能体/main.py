from agent import Agent
from tools import get_web_code, revert_pound_to_kg, revert_meter_to_cm, compute_bmi, get_city_weather

def main():
    agent = Agent()

    agent.register_tool(get_web_code)
    agent.register_tool(revert_pound_to_kg)
    agent.register_tool(revert_meter_to_cm)
    agent.register_tool(compute_bmi)
    agent.register_tool(get_city_weather)

    for schema in agent.tool_schemas:
        print(f"  🛠️  {schema['function']['name']}: {schema['function']['description']}")

    # agent.run("用一句话总结https://www.qiumiwu.com/player/maikeerqiaodan的内容，并且获取成都的天气情况")
    agent.run("用一句话总结https://www.qiumiwu.com/player/maikeerqiaodan的内容")
    agent.run("获取成都的天气情况")
    

if __name__ == "__main__":
    main()