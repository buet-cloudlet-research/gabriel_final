#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#
#   Author: Junjue Wang <junjuew@cs.cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import multiprocessing
import Queue
from optparse import OptionParser
import os
import pprint
import struct
import sys
import time
import pdb

dir_file = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_file, "../.."))

import gabriel
import gabriel.proxy
import json
import cv2
import numpy as np
import base64

LOG = gabriel.logging.getLogger(__name__)
ANDROID_CLIENT = True


def process_command_line(argv):
    VERSION = 'gabriel proxy : %s' % gabriel.Const.VERSION
    DESCRIPTION = "Gabriel cognitive assistance"

    parser = OptionParser(usage='%prog [option]', version=VERSION,
                          description=DESCRIPTION)

    parser.add_option(
        '-s', '--address', action='store', dest='address',
        help="(IP address:port number) of directory server")
    settings, args = parser.parse_args(argv)
    if len(args) >= 1:
        parser.error("invalid arguement")

    if hasattr(settings, 'address') and settings.address is not None:
        if settings.address.find(":") == -1:
            parser.error("Need address and port. Ex) 10.0.0.1:8081")
    return settings, args


class FaceDetectionVideoApp(gabriel.proxy.CognitiveProcessThread):
    object_recognition_created = False
    object_recognition = None

    def __init__(self, *args, **kwargs):
        super(FaceDetectionVideoApp, self).__init__(*args, **kwargs)

        self.face_cascade = cv2.CascadeClassifier(os.path.join(dir_file, "haarcascades", "haarcascade_frontalface_default.xml"))

    def handle(self, header, data):
        # PERFORM Cognitive Assistance Processing
        LOG.info("processing: ")
        LOG.info("%s\n" % header)
        np_data = np.fromstring(data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        rtn_result_dict = {
            "meta":{
                "engine": "face_detection",
                "box_type": "rectangle"
            },
            "results": []
        }

        print faces

        for (x, y, w, h) in faces:

            print x.astype(str)
            print y
            print w
            print type(h)

            ptn1 = {
                "x": x.astype(str),
                "y": y.astype(str)
            }
            ptn2 = {
                "x": (x + w).astype(str),
                "y": (y + h).astype(str)
            }
            rect_spec = [ptn1, ptn2]
            rtn_result_dict["results"].append(rect_spec)
        pass

        rtn_result = json.dumps(rtn_result_dict)

        header[gabriel.Protocol_result.JSON_KEY_STATUS] = 'success'
        header[gabriel.Protocol_result.JSON_KEY_IMAGE] = (0, len(rtn_result))

        print rtn_result

        return rtn_result

if __name__ == "__main__":
    result_queue = multiprocessing.Queue()
    print result_queue._reader

    settings, args = process_command_line(sys.argv[1:])
    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    video_ip = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_IP)
    video_port = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
    ucomm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_IP)
    ucomm_port = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_PORT)

    # image receiving and processing threads
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE  # TODO

    video_receive_client = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_receive_client.start()
    video_receive_client.isDaemon = True

    face_detection_video_app = FaceDetectionVideoApp(image_queue, result_queue, engine_id='face_detection')
    face_detection_video_app.start()
    face_detection_video_app.isDaemon = True

    # result publish
    result_pub = gabriel.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
    finally:
        if video_receive_client is not None:
            video_receive_client.terminate()
        if face_detection_video_app is not None:
            face_detection_video_app.terminate()

        result_pub.terminate()

