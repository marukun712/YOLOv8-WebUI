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

#物体検出
def detect(input,checkbox,conf,iou):
    option = return_options(checkbox)

    #物体検出を実行
    res = detect_model(input,conf=conf,iou=iou,**option)

    #結果を描画
    res_plotted = res[0].plot()
    return res_plotted

#インスタンスセグメンテーション
def seg(input,checkbox,conf,iou):
    option = return_options(checkbox)

    #インスタンスセグメンテーションを実行
    res = seg_model(input,conf=conf,iou=iou,**option)

    #結果を描画
    res_plotted = res[0].plot()
    return res_plotted

#画像分類
def cls(input,checkbox,conf,iou):
    option = return_options(checkbox)

    #画像分類を実行
    res = cls_model(input,conf=conf,iou=iou,**option)

    #結果を描画
    res_plotted = res[0].plot()
    return res_plotted

#姿勢推定
def pose(input,checkbox,conf,iou):
    option = return_options(checkbox)

    #姿勢推定を実行
    res = pose_model(input,conf=conf,iou=iou,**option)

    #結果を描画
    res_plotted = res[0].plot()
    return res_plotted

with gr.Blocks() as app:
    #ヘッダー
    gr.Markdown("# YOLOv8-WebUI")

    #タブ
    with gr.Tabs():
        #Detectタブ
        with gr.TabItem("Detect"):
            with gr.Row():
                detect_input = gr.Image()
                detect_output = gr.Image()
            detect_button = gr.Button("inference")

        #Segmentタブ
        with gr.TabItem("Segment"):
            with gr.Row():
                seg_input = gr.Image()
                seg_output = gr.Image()
            seg_button = gr.Button("inference")

        #Classifyタブ
        with gr.TabItem("Classify"):
            with gr.Row():
                cls_input = gr.Image()
                cls_output = gr.Image()
            cls_button = gr.Button("inference")

        #Poseタブ
        with gr.TabItem("Pose"):
            with gr.Row():
                pose_input = gr.Image()
                pose_output = gr.Image()
            pose_button = gr.Button("inference")

    #オプション
    conf = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.01, interactive=True,label="conf")
    iou = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, interactive=True,label="iou")
    checkbox = gr.CheckboxGroup(["half","show","save","save_txt","save_conf","save_crop","hide_labels","hide_conf","vid_stride","visualize","augment","agnostic_nms","retina_masks","boxes"], label="Options",value=["boxes"])

    #detect_inputから画像を取得してdetect関数を実行
    detect_button.click(detect, inputs=[detect_input,checkbox,conf,iou], outputs=detect_output)

    #seg_inputから画像を取得してdetect関数を実行
    seg_button.click(seg, inputs=[seg_input,checkbox,conf,iou], outputs=seg_output)

    #cls_inputから画像を取得してdetect関数を実行
    cls_button.click(cls, inputs=[cls_input,checkbox,conf,iou], outputs=cls_output)

    #pose_inputから画像を取得してpose関数を実行
    pose_button.click(pose, inputs=[pose_input,checkbox,conf,iou], outputs=pose_output)

#起動
app.launch()