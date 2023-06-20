import importlib.util
import pip
import sys

#パッケージがインストールされているかの確認
def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None

def start():
    print(f"Python {sys.version}")

    print("###################################################################")

    #パッケージがインストールされていなければパッケージをインストールする
    if not (is_installed("gradio")):

        print("Installing gradio")
        pip.main(['install','gradio'])

    if not (is_installed("ultralytics")):

        print("Installing ultralytics")
        pip.main(['install','ultralytics'])

    import app 
    
    #appを起動
    app.app.launch()

start()