import gradio as gr
from ultralytics import YOLO

#各モデルをロード
detect_model = YOLO("yolov8l.pt")
seg_model = YOLO('yolov8n-seg.pt')
cls_model = YOLO('yolov8n-cls.pt')
pose_model = YOLO("yolov8n-pose.pt")

#オプションリストをオブジェクトに整形
def return_options(checkbox):
    option = {}

    #整形
    for check in checkbox:
        option[check] = True

    return option

#結果を描画する
def plot(res):
    plot = res[0].plot()
    return plot

def inference(type,input,checkbox,conf,iou,device,max_det,line_width,cpu):  
    if(type == "detect"):
        model = detect_model
    elif(type == "seg"):
        model = seg_model
    elif(type == "cls"):
        model = cls_model
    elif(type == "pose"):
        model = pose_model

    if(cpu):
        device = "cpu"

    if not(line_width):
        line_width = None

    option = return_options(checkbox)
    
    #物体検出を実行
    res = model(input,conf=conf,iou=iou,device=device,max_det=max_det,line_width=line_width,**option)

    #結果を描画
    plotted = plot(res)
    return plotted

with gr.Blocks() as app:
    #ヘッダー
    gr.Markdown("# YOLOv8-WebUI")

    #タブ
    with gr.Tabs():
        #inferenceタブ
        with gr.TabItem("inference"):
            with gr.Row():
                input = gr.Image()
                output = gr.Image()

            type = gr.Radio(["detect", "seg", "cls", "pose"], value="detect", label="Tasks")

            #オプション 
            conf = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01, interactive=True,label="conf")
            iou = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, interactive=True,label="iou")
            checkbox = gr.CheckboxGroup(["half","show","save","save_txt","save_conf","save_crop","hide_labels","hide_conf","vid_stride","visualize","augment","agnostic_nms","retina_masks","boxes"], label="Options",value=["boxes"])

            device = gr.Number(value=0, label="device", interactive=True, precision=0)
            cpu = gr.Checkbox(label="cpu", interactive=True)
            max_det = gr.Number(value=300, label="max_det", interactive=True, precision=0)
            line_width = gr.Number(value=0, label="line_width", interactive=True, precision=0)

            inference_button = gr.Button("inference")

        with gr.TabItem("train"):
            gr.Markdown("# TODO")

    #inputから画像を取得してdetect関数を実行
    inference_button.click(inference, inputs=[type,input,checkbox,conf,iou,device,max_det,line_width,cpu], outputs=output)
