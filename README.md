# ChatGLM Voice Assistant

## 项目简介
ChatGLM Voice Assistant 是一个基于语音识别和 ChatGLM 模型的语音助手应用。用户可以通过语音输入与 ChatGLM 进行交互，获取智能回复。该项目支持多种语言和主题样式，并提供实时音频波形和频谱可视化。

## 功能特性
- **语音识别**：通过 Google Speech Recognition API 识别用户语音。
- **智能对话**：集成 ChatGLM 模型，提供智能回复。
- **实时音频可视化**：显示音频波形和频谱。
- **多语言支持**：支持英语、中文、西班牙语、法语、德语等。
- **主题切换**：提供多种主题样式（如 Hacker Style、Modern Style 等）。

## 安装步骤

### 1. 克隆项目
```bash
git clone <项目地址>
cd AITool
```

### 2. 安装依赖
确保已安装 Python 3.8 或更高版本，然后运行以下命令：
```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥
在项目根目录创建一个 `.env` 文件，并添加以下内容：
```
ZHIPU_API_KEY=你的API密钥
```

### 4. 运行程序
```bash
python AIspeech.py
```

## 使用方法
1. 启动程序后，选择语言和主题样式。
2. 点击“Start Listening”按钮开始语音识别。
3. 通过麦克风输入语音，程序会将识别的文本发送到 ChatGLM 并显示回复。
4. 点击“Stop Listening”按钮停止语音识别。

## 文件结构
```
AITool/
├── AIspeech.py       # 主程序文件
├── README.md         # 项目说明文档
└── requirements.txt  # 依赖文件
```

## 依赖
- Python 3.8+
- PyQt5
- PyQtGraph
- SpeechRecognition
- Pyaudio
- ZhipuAI
- Numpy
- Dotenv

## 注意事项
- 请确保麦克风设备正常工作。
- 如果语音识别失败，请检查网络连接或 API 配置。

## 贡献
欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证
本项目采用 MIT 许可证。
