# coding=utf-8
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from utils import non_max_suppression, FPSHandler, plot_skeleton_kpts

import deep

import open3d as o3d
from utils_3d import create_segment, coordinates_dynamic_grid, is_present, determine_color, create_kpts_xyd

LINES_BODY = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7], [12, 13], [6, 8],
              [8, 10], [6, 12], [12, 14], [14, 16], [7, 9], [7, 13], [9, 11],  [13, 15], [15, 17]]

nc=80 #80或1

ROOT = Path(__file__).parent

def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y
blob = './yolov8n-pose.blob'
# blob = "/home/mulong/codes/models/yolov8/yolov8n-pose.blob"
model = dai.OpenVINO.Blob(blob)
#print(model.)
dim = next(iter(model.networkInputs.values())).dims
nnWidth, nnHeight = dim[:2]
print(nnWidth, nnHeight)
print(blob)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.NeuralNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("image")
xoutNN.setStreamName("detections")

    # Properties
camRgb.setPreviewSize(nnWidth, nnHeight)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)
camRgb.setPreviewKeepAspectRatio(False)

# Network specific settings
detectionNetwork.setBlob(model)

# Linking
camRgb.preview.link(detectionNetwork.input)
camRgb.preview.link(xoutRgb.input)
# detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(xoutNN.input)

    #########################################
monoLeft = deep.getMonoCamera(pipeline, isLeft=True)
monoRight = deep.getMonoCamera(pipeline, isLeft=False)

    # Combine left and right cameras to form a stereo pair
stereo = deep.getStereoPair(pipeline, monoLeft, monoRight)

xoutDisp = pipeline.createXLinkOut()
xoutDisp.setStreamName("disparity")

xoutDepth=pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")

xoutRectifiedLeft = pipeline.createXLinkOut()
xoutRectifiedLeft.setStreamName("rectifiedLeft")

xoutRectifiedRight = pipeline.createXLinkOut()
xoutRectifiedRight.setStreamName("rectifiedRight")

stereo.disparity.link(xoutDisp.input)
stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
stereo.rectifiedRight.link(xoutRectifiedRight.input)
stereo.depth.link(xoutDepth.input)
#stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
#stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

X1=640
Y1=640
X2=640
Y2=640
second_point=False
mouseX = 0
mouseY = 640
pose_num=0

low_point=None

flag_stright = False
flag_up = False

time_unstright=[time.time(),False]
time_shake=[time.time(),False]
filter_unstright=[time.time(),False]


def run():
    # Connect to device and start pipeline

    with dai.Device(pipeline) as device:
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        imageQueue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
        detectQueue = device.getOutputQueue(
            name="detections", maxSize=4, blocking=False
        )

        frame = None
        detections = []

        fpsHandler = FPSHandler()

        def is_stright(xyvd):
            global flag_stright,low_point,filter_unstright
            try:
                if xyvd[10][3] != -1 and xyvd[8][3] != -1 and xyvd[6][3] != -1 and xyvd[9][3] != -1 and xyvd[7][
                    3] != -1 and xyvd[5][3] != -1:
                    if np.arctan((xyvd[10][1]-xyvd[8][1])/(xyvd[10][0]-xyvd[8][0]))+np.arctan((xyvd[8][1]-xyvd[6][1])/(xyvd[8][0]-xyvd[6][0]))>0.84 and np.arctan((xyvd[9][1]-xyvd[7][1])/(xyvd[9][0]-xyvd[7][0]))+np.arctan((xyvd[7][1]-xyvd[5][1])/(xyvd[7][0]-xyvd[5][0]))<-0.84:
                        if xyvd[10][1]<xyvd[6][1] and xyvd[9][1]<xyvd[5][1]:
                            flag_stright=True
                            print("zhile")
                            if low_point is None:
                                low_point=(xyvd[13][1]+xyvd[14][1])/2
                            else:
                                low_point=min(low_point,(xyvd[13][1]+xyvd[14][1])/2)

            except:
                pass

        def is_up(xiaba):
            global Y1,flag_up
            if xiaba[1]<Y1/2:
                print(Y1/2,xiaba)
                flag_up=True
        def count_plus(flagup,flagstraight,frame):
            global pose_num,flag_stright,flag_up,time_unstright
            if flagup == True and flagstraight == False:
                flag_up = False     #这俩是用来衡量是否做一个动作的
                flag_stright = False

                time_unstright[0]=time.time()   #这个是计时器【时间，是否展示】
                time_unstright[1]=True

            elif flagup and flagstraight:
                pose_num+=1
                flag_up = False
                flag_stright = False
                filter_unstright[0]=time.time()
                filter_unstright[1]=True
            return frame

        def is_shake(xyvd):
            global time_shake
            try:
                if xyvd[10][3] != -1 and xyvd[8][3] != -1 and xyvd[6][3] != -1 and xyvd[9][3] != -1 and xyvd[7][
                    3] != -1 and xyvd[5][3] != -1:
                    if np.arctan((xyvd[10][1]-xyvd[8][1])/(xyvd[10][0]-xyvd[8][0]))+np.arctan((xyvd[8][1]-xyvd[6][1])/(xyvd[8][0]-xyvd[6][0]))>0.84 and np.arctan((xyvd[9][1]-xyvd[7][1])/(xyvd[9][0]-xyvd[7][0]))+np.arctan((xyvd[7][1]-xyvd[5][1])/(xyvd[7][0]-xyvd[5][0]))<-0.84:

                        if abs(xyvd[13][1]-low_point)>0.6*abs(xyvd[13][1]-xyvd[14][1]) or abs(xyvd[14][1]-low_point)>0.6*abs(xyvd[14][1]-xyvd[16][1]):
                            if xyvd[10][1] < xyvd[6][1] and xyvd[9][1] < xyvd[5][1]:
                                time_shake[0]=time.time()
                                time_shake[1]=True
                                print(low_point,xyvd[13][1],xyvd[14][1])
            except:
                pass

        def show_message(frame):
            global time_shake,time_unstright,pose_num
            if time.time()-time_unstright[0]<5 and time_unstright[1]==True and filter_unstright[1]==False:#没伸直显示
                cv2.putText(frame, 'underrock', (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 100), 1, cv2.LINE_AA
                            )
            elif time.time()-time_unstright[0]<5 and time_unstright[1]==True and filter_unstright[1]==True:
                if time.time()-filter_unstright[0]>2:
                    filter_unstright[1]=False
                    time_unstright[1]=False
            else:
                time_unstright[1] = False


            if time.time()-time_shake[0]<2 and time_shake[1]==True:         #震荡显示
                cv2.putText(frame, 'shake', (130, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 100), 1, cv2.LINE_AA
                            )
            else:
                time_shake[1] = False

            cv2.putText(frame, str(pose_num), (255, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )

            return frame


        def drawText(frame, text, org, color=(255, 255, 255)):
            cv2.putText(
                frame,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        def displayFrame(name, frame, depth):######################################主要都加在这里了
            global X1,Y1,X2,Y2,second_point,pose_num
            color = (255, 0, 0)
            xyvd=None
            for detection in detections:#detections 是检测到的人体个数
                *bbox, conf, cls = detection[:6]#57个 类别 置信度 bbox 17*3
                bbox = np.array(bbox).astype(int)
                kpts = detection[6:]
                #print(kpts)
                #print(kpts.shape)

                # if int(cls) == 0:
                drawText(
                    frame,
                    f"{conf:.2%}",
                    (bbox[0] + 10, bbox[1] + 35),
                )
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
                )
                frame,xyvd = plot_skeleton_kpts(frame, kpts=kpts, steps=3,depth=depth)
            if xyvd is not None:
                #xiaba=[(xyvd[1][0]+xyvd[2][0])/2,(xyvd[1][1]+xyvd[2][1])/2-4*((xyvd[1][1]+xyvd[2][1])/2-xyvd[0][1])]
                xiaba = [(xyvd[1][0] + xyvd[2][0]) / 2,
                         0.6*((xyvd[5][1]+xyvd[6][1])/2-xyvd[0][1])+xyvd[0][1]]

                frame = cv2.circle(frame, (int(xiaba[0]), int(xiaba[1])), 2,
                                   (255, 255, 100), 2)
                is_up(xiaba)
                is_stright(xyvd)
                is_shake(xyvd)
                frame = count_plus(flag_up, flagstraight=flag_stright, frame=frame)

            frame = show_message(frame)
            frame = cv2.resize(frame, (640, 640))       #resize要放在画点之前，这样点在绘制时才能以640窗口采的位置绘制在640图上
                                                        #这样判断下巴过杠需要Y除2，因为点是以320绘制的，线是以640绘制的

            # Show the frame
            if second_point== False:
                print(mouseX, mouseY)
                frame= cv2.circle(frame, (mouseX, mouseY), 2,
                                   (255, 255, 128), 2)
                X1,Y1=mouseX,mouseY
                if X1<320 and Y1<320:
                    second_point=True
            else:
                frame = cv2.circle(frame, (X1, Y1), 2,
                                   (255, 255, 128), 2)
                X2,Y2=mouseX,mouseY
                frame = cv2.circle(frame, (X2, Y1), 2,
                                   (255, 255, 128), 2)
                frame = cv2.line(frame, (X1, Y1),
                                 (X2, Y1), (0, 0, 255), 2)



            #
            cv2.imshow(name, frame)

            return frame,xyvd

        def toTensorResult(packet):
            """
            Converts NN packet to dict, with each key being output tensor name and each value being correctly reshaped and converted results array
            Useful as a first step of processing NN results for custom neural networks
            Args:
                packet (depthai.NNData.NNData): Packet returned from NN node
            Returns:
                dict: Dict containing prepared output tensors
            """
            data = {}
            for tensor in packet.getRaw().tensors:
                if tensor.dataType == dai.TensorInfo.DataType.INT:
                    data[tensor.name] = np.array(
                        packet.getLayerInt32(tensor.name)
                    ).reshape(tensor.dims)
                elif tensor.dataType == dai.TensorInfo.DataType.FP16:
                    data[tensor.name] = np.array(
                        packet.getLayerFp16(tensor.name)
                    ).reshape(tensor.dims)
                elif tensor.dataType == dai.TensorInfo.DataType.I8:
                    data[tensor.name] = np.array(
                        packet.getLayerUInt8(tensor.name)
                    ).reshape(tensor.dims)
                else:
                    print("Unsupported tensor layer type: {}".format(tensor.dataType))
            return data

        bboxColors = np.random.randint(255, size=3)



        ###############################################################################
        disparityQueue = device.getOutputQueue(name="disparity",
                                               maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft",
                                                   maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight",
                                                    maxSize=1, blocking=False)
        depthQueue=device.getOutputQueue(name='depth',maxSize=1,blocking=False)

        # Calculate a multiplier for color mapping disparity map

        disparityMultiplier = 255 / stereo.getMaxDisparity()
        #print(disparityMultiplier)

        # Variable use to toggle between side by side view and one frame  view.
        cv2.namedWindow("image")

        cv2.setMouseCallback("image", mouseCallback)

        # 3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d
        # 初始化可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 设置动态效果的参数
        to_reset = True

        point_cloud = o3d.geometry.PointCloud()
        vis.add_geometry(point_cloud)

        # 初始化
        line = create_segment([0, 0, 0], [0, 0, 0], radius=0.005, color=(0, 0, 0))
        vis.add_geometry(line)

        lines = []


        while not device.isClosed():
            batch_bboxes, batch_poses, batch_scores = [], [], []
            imageQueueData = imageQueue.tryGet()
            detectQueueData = detectQueue.tryGet()


            depth = deep.getFrame(depthQueue)
            depth = cv2.resize(depth, (320, 320))
            # print(depth)

            kpts_xyvd = None


            if imageQueueData is not None:
                frame = imageQueueData.getCvFrame()
                fpsHandler.tick("color")

            if detectQueueData is not None:
                pred = toTensorResult(detectQueueData)["output0"]
                fpsHandler.tick("nn")
                detections = non_max_suppression(pred, 0.5, nc=nc)[0]#

            if frame is not None:
                fpsHandler.drawFps(frame, "color")

                kpts_xyvd = displayFrame("image", frame,depth)[1]
#               vid_writer.write(frame)
                frame = None

        ##########################################################################################

            # Get the disparity map.
            disparity =deep.getFrame(disparityQueue)
            #print(disparity.shape)
            disparity=cv2.resize(disparity,(320,320))
            #print(disparity)

            '''depth=deep.getFrame(depthQueue)
            depth = cv2.resize(depth, (320, 320))
            #print(depth)
'''
            # Colormap disparity for display.
            disparity = (disparity *
                             disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            disparity=cv2.resize(disparity,(640,640))

                # Get the left and right rectified frame.


            cv2.imshow("Disparity", disparity)
        ###########################################################################################

            if kpts_xyvd is not None:
                kpts_xyd = create_kpts_xyd(kpts_xyvd)

                object_coordinates = coordinates_dynamic_grid(kpts_xyd)

                # for line in lines:
                #     vis.remove_geometry(line, reset_bounding_box=False)
                vis.clear_geometries()

                lines = []
                color_position = 0

                if object_coordinates is not None:
                    for i, a_b in enumerate(LINES_BODY):
                        a, b = a_b
                        if is_present(kpts_xyd[a - 1]) and is_present(
                                kpts_xyd[b - 1]):
                            # 获取关键点的坐标
                            point_a = object_coordinates[a - 1]
                            point_b = object_coordinates[b - 1]

                            color = determine_color(color_position)

                            # 更新LineSet的坐标
                            line = create_segment(point_a, point_b, radius=0.02, color=color)
                            lines.append(line)

                        color_position += 1

                    for line in lines:
                        vis.add_geometry(line)

                    if to_reset:
                        vis.reset_view_point(True)
                        to_reset = False

                    vis.poll_events()
                    vis.update_renderer()

                    vis.remove_geometry(line, reset_bounding_box=False)

            else:
                vis.remove_geometry(line, reset_bounding_box=False)

            if cv2.waitKey(1) == ord("q"):
                #vid_writer.release()
                break

if __name__ == "__main__":
    from loguru import logger

    with logger.catch():
        run()

