from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import os
import argparse
import keyboard


# standard argparse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=4,
    help='use -d to change the distance of the drone. Range 0-6')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
    help='use -sx to change the saftey bound on the x axis . Range 0-480')
parser.add_argument('-sy', '--saftey_y', type=int, default=75,
    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-os', '--override_speed', type=int, default=1,
    help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()


#set points (center of the frame coordinates in pixels)
rifX = 960/2
rifY = 720/2

#PI constant
Kp_X = 0.1
Ki_X = 0.0
Kp_Y = 0.2
Ki_Y = 0.0

#Loop time
Tc = 0.05

#PI terms initialized
integral_X = 0
error_X = 0
previous_error_X = 0
integral_Y = 0
error_Y = 0
previous_error_Y = 0

#global centroX_pre
centroX_pre = rifX
#global centroY_pre
centroY_pre= rifY

#neural network 改成网络参数文件位置
net = cv2.dnn.readNetFromCaffe(r"E:\Coding\UAVclass\TT_demo\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master/MobileNetSSD_deploy.prototxt.txt",
							   r"E:\Coding\UAVclass\TT_demo\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master/MobileNetSSD_deploy.caffemodel") #modify with the NN path
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Speed of the drone
S = 20
S2 = 5
UDOffset = 0#

# this is just the bound box sizes that openCV spits out *shrug*
faceSizes = [1026, 684, 456, 304, 202, 136, 90]

# These are the values in which kicks in speed up mode, as of now, this hasn't been finalized or fine tuned so be careful
# Tested are 3, 4, 5
acc = [500,250,250,150,110,70,50]

# Frames per second of the pygame window display
FPS = 25
dimensions = (960, 720)

# 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()

# If we are to save our sessions, we need to make sure the proper directories exist
if args.save_session:
    ddir = "Sessions"

    if not os.path.isdir(ddir):
        os.mkdir(ddir)

    ddir = "Sessions/Session {}".format(str(datetime.datetime.now()).replace(':','-').replace('.','_'))
    os.mkdir(ddir)

class MyDrone(object):
    
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False
        self.is_obstacle_avoid = False
        self.centroX_pre = rifX
        self.centroY_pre = rifY
        # PID parameters:
        self.integral_X = 0
        self.error_X = 0
        self.previous_error_X = 0
        self.integral_Y = 0
        self.error_Y = 0
        self.previous_error_Y = 0


    def run(self):
        '''飞行控制+人体识别
        '''

        #Kp_X = 2
        # if not self.tello.connect():
        #     print("Tello not connected")
        #     return
        #
        # if not self.tello.set_speed(self.speed):
        #     print("Not set speed to lowest possible")
        #     return

        # In case streaming is on. This happens when we quit this program without the escape key.
        # if not self.tello.streamoff():
        #     print("Could not stop video stream")
        #     return
        #
        # if not self.tello.streamon():
        #     print("Could not start video stream")
        #     return
        #self.tello.connect()
        self.tello.set_speed(self.speed)
        #self.tello.streamoff()
        self.tello.streamon()

        self.tello.enable_mission_pads()
        self.tello.set_mission_pad_detection_direction(0)  # 下视模式

        self.tello.send_expansion_command("mled s b 7")
        frame_read = self.tello.get_frame_read()

        should_stop = False
        imgCount = 0
        OVERRIDE = False
        oSpeed = args.override_speed
        tDistance = args.distance
        #self.tello.get_battery()
        
        # Safety Zone X
        szX = args.saftey_x # 100

        # Safety Zone Y
        szY = args.saftey_y # 55
        
        if args.debug:
            print("DEBUG MODE ENABLED!")

        while not should_stop:
            #self.update() # 更新速度

            if frame_read.stopped:
                frame_read.stop()
                break

            theTime = str(datetime.datetime.now()).replace(':','-').replace('.','_')

            #frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = frame_read.frame
            #frameRet = frame_read.frame

            #vid = self.tello.get_video_capture()

            #if args.save_session:
            #    cv2.imwrite("{}/tellocap{}.jpg".format(ddir,imgCount),frameRet)
            imgCount+=1

            # Listen for key presses
            k = cv2.waitKey(20)

            # Press 0 to set distance to 0
            if k == ord('0'):
                if not OVERRIDE:
                    print("Distance = 0")
                    tDistance = 0

            # Press 1 to set distance to 1
            if k == ord('1'):
                if OVERRIDE:
                    oSpeed = 1
                else:
                    print("Distance = 1")
                    tDistance = 1

            # Press 2 to set distance to 2
            if k == ord('2'):
                if OVERRIDE:
                    oSpeed = 2
                else:
                    print("Distance = 2")
                    tDistance = 2
                    
            # Press 3 to set distance to 3
            if k == ord('3'):
                if OVERRIDE:
                    oSpeed = 3
                else:
                    print("Distance = 3")
                    tDistance = 3
            
            # Press 4 to set distance to 4
            if k == ord('4'):
                if not OVERRIDE:
                    print("Distance = 4")
                    tDistance = 4
                    
            # Press 5 to set distance to 5
            if k == ord('5'):
                if not OVERRIDE:
                    print("Distance = 5")
                    tDistance = 5
                    
            # Press 6 to set distance to 6
            if k == ord('6'):
                if not OVERRIDE:
                    print("Distance = 6")
                    tDistance = 6

            # Press T to take off
            if k == ord('t'):
                if not args.debug:
                    print("Taking Off")
                    self.tello.takeoff()
                    self.tello.move_up(50)
                    self.battery()
                self.send_rc_control = True

            # Press L to land
            if k == ord('l'):
                if not args.debug:
                    print("Landing")
                    self.tello.land()
                self.send_rc_control = False

            if k == ord('b'):
                if not args.debug:
                    print("Now broadcasting!")
                    self.broadcast()

            # Press Backspace(<-) for controls override
            if k == 8:
                if not OVERRIDE:
                    OVERRIDE = True
                    print("OVERRIDE ENABLED")
                else:
                    OVERRIDE = False
                    print("OVERRIDE DISABLED")

            if OVERRIDE:# 超控模式，自动模式遇到问题时，人为接管
                self.update()

                # S & W to fly forward & back
                if k == ord('w'):
                    self.for_back_velocity = int(S * oSpeed)
                elif k == ord('s'):
                    self.for_back_velocity = -int(S * oSpeed)
                else:
                    self.for_back_velocity = 0

                # a & d to pan left & right
                if k == ord('d'):
                    self.yaw_velocity = int(S * oSpeed)
                elif k == ord('a'):
                    self.yaw_velocity = -int(S * oSpeed)
                else:
                    self.yaw_velocity = 0

                # Q & E to fly up & down
                if k == ord('e'):
                    self.up_down_velocity = int(S * oSpeed)
                elif k == ord('q'):
                    self.up_down_velocity = -int(S * oSpeed)
                else:
                    self.up_down_velocity = 0

                # c & z to fly left & right
                if k == ord('c'):
                    self.left_right_velocity = int(S * oSpeed)
                elif k == ord('z'):
                    self.left_right_velocity = -int(S * oSpeed)
                else:
                    self.left_right_velocity = 0

                if self.tello.get_mission_pad_id() == 2:
                    self.tello.go_xyz_speed_mid(0, 0, 100, 30, 1)
                    self.tello.land()
                    break

            # press "ESC" to quit the software
            if k == 27:
                should_stop = True
                break
            # Target size
            #tSize = faceSizes[tDistance]

            # These are our center dimensions
            #cWidth = int(dimensions[0]/2)
            #cHeight = int(dimensions[1]/2)

            #noFaces = len(faces) == 0

            # if we've given rc controls & get face coords returned
            if self.send_rc_control and not OVERRIDE:
                '''************************************************'''
                start = time.time()
                frame = self.tello.get_frame_read().frame

                cv2.circle(frame, (int(rifX), int(rifY)), 1, (0, 0, 255), 10)

                h, w, channels = frame.shape

                blob = cv2.dnn.blobFromImage(frame,
                                             0.007843, (180, 180), (0, 0, 0), True, crop=False)

                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):

                    idx = int(detections[0, 0, i, 1])
                    confidence = detections[0, 0, i, 2]

                    if CLASSES[idx] == "person" and confidence > 0.5:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      colors[idx], 2)
                        # draw the center of the person detected
                        centroX = (startX + endX) / 2
                        centroY = (2 * startY + endY) / 3

                        self.centroX_pre = centroX
                        self.centroY_pre = centroY

                        cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

                        self.error_X = -(rifX - centroX)
                        self.error_Y = rifY - centroY

                        cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

                        integral_X = integral_X + error_X * Tc

                        # PI controller
                        self.integral_X = self.integral_X + self.error_X * Tc  # updating integral PID term
                        uX = Kp_X * self.error_X + Ki_X * self.integral_X  # updating control variable uX
                        self.previous_error_X = self.error_X  # update previous error variable

                        self.integral_Y = self.integral_Y + self.error_Y * Tc  # updating integral PID term
                        uY = Kp_Y * self.error_Y + Ki_Y * self.integral_Y
                        self.previous_error_Y = self.error_Y

                        X_1 = abs(endX - startX)
                        Y_1 = abs(endY - startY)
                        scale_1 = (X_1 * Y_1) / (960 * 720)
                        if scale_1 > 0.3: # TODO: tune the parameter to change distance of tracking
                            speed_1 = -20
                        elif scale_1 < 0.2:
                            speed_1 = 20
                        else:
                            speed_1 = 0

                        self.for_back_velocity = speed_1
                        self.up_down_velocity = round(uY)
                        self.yaw_velocity = round(uX)
                        #print(self.up_down_velocity,'\n')
                        #print(self.yaw_velocity,'\n')
                        self.update()
                        #drone.send_rc_control(0, 0, round(uY), round(uX))
                        # break when a person is recognized

                        break


                    else:  # if nobody is recognized take as reference centerX and centerY of the previous frame
                        centroX = self.centroX_pre
                        centroY = self.centroY_pre
                        cv2.circle(frame, (int(centroX), int(centroY)), 1, (0, 0, 255), 10)

                        self.error_X = -(rifX - centroX)
                        self.error_Y = rifY - centroY

                        cv2.line(frame, (int(rifX), int(rifY)), (int(centroX), int(centroY)), (0, 255, 255), 5)

                        self.integral_X = self.integral_X + self.error_X * Tc  # updating integral PID term
                        uX = Kp_X * self.error_X + Ki_X * self.integral_X  # updating control variable uX
                        self.previous_error_X = self.error_X  # update previous error variable

                        self.integral_Y = self.integral_Y + self.error_Y * Tc  # updating integral PID term
                        uY = Kp_Y * self.error_Y + Ki_Y * self.integral_Y
                        self.previous_error_Y = self.error_Y

                        self.for_back_velocity = 0
                        self.up_down_velocity = round(uY)
                        self.yaw_velocity = round(uX)
                        self.update()
                        #drone.send_rc_control(0, 0, round(uY), round(uX))

                        continue

                #cv2.imshow("Frame", frame)
                # 存储视频
                #video.write(frame)

                end = time.time()
                elapsed = end - start
                if Tc - elapsed > 0:
                    time.sleep(Tc - elapsed)
                end_ = time.time()
                elapsed_ = end_ - start
                fps = 1 / elapsed_
                print("FPS: ", fps)

                if cv2.waitKey(1) == 27:  # press 'esc' to quit
                    break
                '''********************************************'''

            dCol = lerp(np.array((0, 0, 255)), np.array((255, 255, 255)), tDistance + 1 / 7)
            if OVERRIDE:
                show = "OVERRIDE: {}".format(oSpeed)
                dCol = (255,255,255)
            else:
                #show = "AI_track_person: {}".format(str(tDistance))
                show = "AI_track_person:"

            # Draw the distance choosen
            cv2.putText(frame,show,(32,664),cv2.FONT_HERSHEY_SIMPLEX,1,dCol,2)

            # Display the resulting frame
            cv2.imshow(f'Tello Tracking...',frame)

        # On exit, print the battery
        self.battery()

        # When everything done, release the capture
        cv2.destroyAllWindows()
        
        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def battery(self):
        print(f'battery : {self.tello.get_battery()}')

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def obstacle_avoid(self):
        # if not self.tello.connect():
        #     print("Tello not connected")
        #     return
        self.tello.connect()
        # if not self.tello.streamoff():
        #     print("Could not stop video stream")
        #     return
        self.tello.send_expansion_command("mled s b 7")
        self.tello.takeoff()
        self.tello.move_up(30)
        self.tello.enable_mission_pads()
        self.tello.set_mission_pad_detection_direction(0)  # 下视模式

        print('start obstacle avoidance:\n')
        while self.is_obstacle_avoid:
            pad = self.tello.get_mission_pad_id()
            if pad == 1:
                # 发现 1 号挑战卡后，飞行至其上空
                self.tello.go_xyz_speed_mid(0, 0, 100, 30, 1)
                self.broadcast() # 播送录音
                self.tello.send_expansion_command("mled l r 1 calm_down!")
                break
            dis = self.tello.send_read_command('EXT tof?')
            if len(dis) < 10:
                dis_num = eval(dis[4:])
                if dis_num < 400:  # 避障
                    self.tello.send_rc_control(-20, 0, 0, 0)
                    self.tello.send_expansion_command("led 255 0 0")
                else:  # 继续动作
                    self.tello.send_rc_control(0, 20, 0, 0)
                    self.tello.send_expansion_command("led 255 255 255")
            if keyboard.is_pressed('m'):
                self.is_obstacle_avoid = False
                self.tello.send_rc_control(0, 0, 0, 0)
                break

        print('ending obstacle avoidance...\n')
        self.tello.disable_mission_pads()
        self.tello.land()
        #self.tello.end()

    def broadcast(self):
        file = 'resque.wav'
        print('-----Now broadcasting-----')
        os.system(file)


def lerp(a,b,c):
    return a + c*(b-a)

def main():
    drone_1 = MyDrone()
    drone_1.is_obstacle_avoid = True
    # run drone_1
    print('start obstacle_avoid:\n')
    drone_1.obstacle_avoid()
    print('start tracking:\n')
    drone_1.run()
    #drone_1.broadcast()



if __name__ == '__main__':
    main()
