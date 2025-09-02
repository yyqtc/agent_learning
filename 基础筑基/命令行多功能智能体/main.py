from agent import Agent
from tools import get_web_code, revert_pound_to_kg, revert_meter_to_cm, compute_bmi

def main():
    agent = Agent()

    agent.register_tool(get_web_code)
    agent.register_tool(revert_pound_to_kg)
    agent.register_tool(revert_meter_to_cm)
    agent.register_tool(compute_bmi)

    for schema in agent.tool_schemas:
        print(f"  🛠️  {schema['function']['name']}: {schema['function']['description']}")

    agent.run("请总结https://www.qiumiwu.com/player/maikeerqiaodan的内容，如果可以的话，我的身高177，体重87.7公斤，计算我的BMI")
    

if __name__ == "__main__":
    main()