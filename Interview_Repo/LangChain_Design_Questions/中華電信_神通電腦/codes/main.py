def main():
    agent = setup_agent()
    
    print("AI Assistant 已啟動，請說出您的需求...")
    
    while True:
        user_input = input("您: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            break
            
        try:
            response = agent.run(user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"發生錯誤: {str(e)}")
            
if __name__ == "__main__":
    main()
