import torch
import emotion  # 导入你之前修改好的模型模块

def main():
    # 1. 初始化模型参数（必须与训练时保持一致）
    VOCAB_SIZE = len(emotion.vocab)
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1

    # 2. 加载模型并切换到评估模式
    loaded_model = emotion.SimpleLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    loaded_model.load_state_dict(torch.load('sentiment_model.pth'))
    loaded_model.eval()

    print("欢迎使用情感分析交互系统！")
    print("提示: 输入任意英文句子进行情感预测，输入 'quit' 或 'exit' 退出程序。\n")

    # 3. 开启无限循环接收用户输入
    while True:
        try:
            # 获取用户输入，并去除首尾空格
            sentence = input("请输入句子: ").strip()
            
            # 4. 退出条件判断
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            # 5. 防止用户输入空字符串
            if not sentence:
                print("⚠️ 输入不能为空，请重新输入。\n")
                continue

            # 6. 进行预测
            with torch.no_grad():
                text_tensor = torch.tensor([emotion.text_pipeline(sentence)], dtype=torch.long)
                prediction = loaded_model(text_tensor).item()
                sentiment = "正面" if prediction > 0.5 else "负面"
                
            print(f"预测结果: 【{sentiment}】 (置信度: {prediction:.4f})\n")
            
        # 7. 处理强制退出（如按下 Ctrl+C）
        except KeyboardInterrupt:
            print("\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}\n")

if __name__ == '__main__':
    main()