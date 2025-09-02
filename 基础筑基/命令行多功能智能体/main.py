from agent import Agent
from tools import get_web_code, revert_pound_to_kg, revert_meter_to_cm, compute_bmi, get_city_weather

def main():
    # 初始化智能体
    agent = Agent()

    # 注册工具
    agent.register_tool(get_web_code)
    agent.register_tool(revert_pound_to_kg)
    agent.register_tool(revert_meter_to_cm)
    agent.register_tool(compute_bmi)
    agent.register_tool(get_city_weather)

    # 显示欢迎信息和可用工具
    print("智能体系统已启动！")
    print("可用工具:")
    for schema in agent.tool_schemas:
        print(f"{schema['function']['name']}: {schema['function']['description']}")
    print("\n输入 'exit' 退出系统")
    print("=" * 50)

    # 主循环：等待用户输入
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的需求: ").strip()
            
            # 检查退出条件
            if user_input.lower() == 'exit':
                print("\n感谢使用智能体系统，再见！")
                break
            
            # 检查空输入
            if not user_input:
                print("请输入有效内容")
                continue
            
            # 处理用户输入
            print(f"\n正在处理: {user_input}")
            print("-" * 30)
            
            try:
                # 调用智能体处理用户输入
                agent.run(user_input)
            except Exception as e:
                print(f"处理过程中出现错误: {str(e)}")
            
            print("-" * 30)
            print("处理完成，等待下一个输入...")
            
        except KeyboardInterrupt:
            # 处理 Ctrl+C 中断
            print("\n\n检测到中断信号，系统正在退出...")
            break
        except EOFError:
            # 处理 EOF 错误（如 Ctrl+D）
            print("\n\n输入结束，系统正在退出...")
            break
        except Exception as e:
            print(f"\n系统错误: {str(e)}")
            print("系统将继续运行，请重新输入...")

if __name__ == "__main__":
    main()