import os

# Configure local cache at the very beginning
os.environ['TORCH_HOME'] = os.path.abspath('.cache/torch')
os.environ['HF_HOME'] = os.path.abspath('.cache/huggingface')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ui.app import App

if __name__ == "__main__":
    print("Initializing Video-to-Funscript Tool...")
    app = App()
    app.mainloop()
