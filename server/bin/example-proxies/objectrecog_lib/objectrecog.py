#!/usr/bin/env python
import inspect
import os
import sys
import time
import traceback

import cv2
import numpy as np

import utilscv

dir_file = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_file, "../../.."))
import gabriel
import gabriel.proxy

LOG = gabriel.logging.getLogger(__name__)

MIN_MATCH_COUNT = 20
MIN_INLIER = 5

# "ALL" for comparing all image
# "ODD" for odd image
# "EVEN" for even image
CONFIG_COMP = "ALL"


# If you want to make a pile of Python, please ask ImageFeature
# If there is a case of unauthorized access to the database,
# Reconciliation of the information provided to the information provider
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        # Nombre del fichero
        self.nameFile = nameFile
        # Shape de la imagen
        self.shape = shape
        # Datos binarios de la imagen
        self.imageBinary = imageBinary
        # Keypoints of the image once applied the feature detection algorithm
        self.kp = kp
        # Descriptores de las features detectadas
        self.desc = desc
        # Matching of the image of the database with the image of the webcam
        self.matchingWebcam = []
        # Matching the webcam with the current image of the database
        self.matchingDatabase = []

    # It allows to empty the calculated calculations previously, for a new image
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []


def nothing(*arg):
    pass


def raw2cv_image(raw_data):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    cv_image = cv2.imdecode(img_array, -1)
    return cv_image


def print_frame():
    callerframerecord = inspect.stack()[1]  # 0 represents this line # 1 represents line at caller

    frame = callerframerecord[0]

    info = inspect.getframeinfo(frame)
    # print info.filename                       # __FILE__     -> Test.py
    # print info.function                       # __FUNCTION__ -> Main
    # print info.lineno                         # __LINE__     -> 13
    return info.lineno


class ObjectRecognition:
    log_images_counter = 0

    def __init__(self, method=4):
        self.log_video_writer_created = False
        self.log_video_writer = None

        LOG.info("ObjectRecognition Class initializing...")

        # Creating window and associated sliders, and mouse callback:
        cv2.namedWindow('Features')
        cv2.namedWindow('ImageDetector')

        # Selection of the method to compute the features
        cv2.createTrackbar('method', 'Features', 0, 4, nothing)
        # Playback error for calculating inliers with RANSAC
        cv2.createTrackbar('projer', 'Features', 5, 10, nothing)
        # Minimum number of inliers to indicate that an object has been recognized
        cv2.createTrackbar('inliers', 'Features', 20, 50, nothing)
        # Trackbar to indicate whether features are painted or not
        cv2.createTrackbar('drawKP', 'Features', 0, 1, nothing)

        # Creation of the feature detector, according to method (only at the beginning):
        # self.method = cv2.getTrackbarPos('method', 'Features')
        self.method = method
        self.method_str = ''

        if self.method == 0:
            if self.method_str != 'SIFT':
                # no sift error kill
                self.method_str = 'SIFT'
                # The number of features has been limited to 250 for the algorithm to flow.
                self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=250)
        elif self.method == 1:
            if self.method_str != 'AKAZE':
                self.method_str = 'AKAZE'
                self.detector = cv2.AKAZE_create()
        elif self.method == 2:
            if self.method_str != 'SURF':
                self.method_str = 'SURF'
                self.detector = cv2.xfeatures2d.SURF_create(800)
        elif self.method == 3:
            if self.method_str != 'ORB':
                self.method_str = 'ORB'
                self.detector = self.orb = cv2.ORB_create(400)
        elif self.method == 4:
            if self.method_str != 'BRISK':
                self.method_str = 'BRISK'
                if hasattr(cv2, "BRISK_create"):
                    self.detector = cv2.BRISK_create()
                else:
                    self.detector = cv2.BRISK()

        self.dataBase = dict([('SIFT', []), ('AKAZE', []), ('SURF', []),
                              ('ORB', []), ('BRISK', [])])

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.load_models_from_directory()

    # Funcion encircled the calcular, for example, the calculation of the features of the calculations,
    # Load features of the imagery of the directory of "models"
    def load_models_from_directory(self):
        # The method returns a dictionary. The key is the features algorithm
        # while the value is a list of objects of type ImageFeature
        # where all the data of the features of the images of the Database are stored

        config_index = 0
        for imageFile in os.listdir(os.path.join(dir_file, "models")):
            if CONFIG_COMP == "ODD" and config_index % 2:
                config_index += 1
                continue
            elif CONFIG_COMP == "EVEN" and not config_index % 2:
                config_index += 1
                continue

            config_index += 1

            # The image is loaded with the OpenCV
            colorImage = cv2.imread(os.path.join(dir_file, "models/" + str(imageFile)))
            # We pass the grayscale image
            currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
            # We perform a resize of the image, so that the compared image is equal

            kp, desc = self.detector.detectAndCompute(currentImage, None)
            self.dataBase[self.method_str].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))

        return self.dataBase

    # Function responsible for calculating mutual matching, but nesting loops
    # It is a very slow solution because it does not take advantage of Numpy power
    # We do not even put a slider to use this method as it is very slow
    def find_matching_mutuos(self, method_str, desc, kp):
        for key, item in enumerate(self.dataBase[method_str]):
            self.dataBase[method_str][key].clearMatchingMutuos()
            for i in range(len(desc)):
                primerMatching = None
                canditatoDataBase = None
                matchingSegundo = None
                candidateWebCam = None
                for j in range(len(self.dataBase[method_str][key].desc)):
                    valorMatching = np.linalg.norm(desc[i] - self.dataBase[method_str][key].desc[j])
                    if (primerMatching is None or valorMatching < primerMatching):
                        primerMatching = valorMatching
                        canditatoDataBase = j
                for k in range(len(desc)):
                    valorMatching = np.linalg.norm(self.dataBase[method_str][key].desc[canditatoDataBase] - desc[k])
                    if (matchingSegundo is None or valorMatching < matchingSegundo):
                        matchingSegundo = valorMatching
                        candidateWebCam = k
                if not candidateWebCam is None and i == candidateWebCam:
                    self.dataBase[method_str][key].matchingWebcam.append(kp[i].pt)
                    self.dataBase[method_str][key].matchingDatabase.append(
                        self.dataBase[method_str][key].kp[canditatoDataBase].pt)
        return self.dataBase[method_str]

    # Function responsible for calculating the mutual matching of a webcam image,
    # With all the images of the database. Receive as input parameter
    # The database based on the method of calculation of features used
    # In the image input of the webcam.
    def find_matching_mutuos_optimum(self, method_str, desc, kp):
        # The algorithm is repeated for each image in the database.
        for key, item in enumerate(self.dataBase[method_str]):
            self.dataBase[method_str][key].clearMatchingMutuos()
            for i in range(len(desc)):
                # The standard of the difference of
                # the current descriptor, with all descriptors of the image of the database, is calculated.
                # We got Without loops and making use of Numpy broadcasting,
                # all distances Between the current descriptor with all descriptors of the current image
                distanceListFromWebCam = np.linalg.norm(desc[i] - self.dataBase[method_str][key].desc, axis=-1)
                # You get the candidate who is the shortest distance from the current descriptor
                candidatoDataBase = distanceListFromWebCam.argmin()
                # It is checked if the matching is mutual, that is,
                # if it is true In the other direction.
                # That is, it is verified that the candidateDatabase Has the current descriptor as best matching
                distanceListFromDataBase = np.linalg.norm(self.dataBase[method_str][key].desc[candidatoDataBase] - desc,
                                                          axis=-1)
                candidatoWebCam = distanceListFromDataBase.argmin()

                # If mutual matching is fulfilled, it is stored for later processing
                if abs(i - candidatoWebCam) < MIN_MATCH_COUNT:
                    self.dataBase[method_str][key].matchingWebcam.append(kp[i].pt)
                    self.dataBase[method_str][key].matchingDatabase.append(
                        self.dataBase[method_str][key].kp[candidatoDataBase].pt)
            # For convenience they become Numpy ND-Array
            self.dataBase[method_str][key].matchingWebcam = np.array(self.dataBase[method_str][key].matchingWebcam)
            self.dataBase[method_str][key].matchingDatabase = np.array(self.dataBase[method_str][key].matchingDatabase)

        return self.dataBase[method_str]

    # This function calculates the best image based on the number of inliers
    # Which has each image of the database with the image obtained from
    # The web camera.
    def calculate_best_image_by_num_inliers(self, method_str, projer, minInliers):
      #  if minInliers < MIN_INLIER:
        minInliers = MIN_INLIER

        bestIndex = None
        bestMask = None
        numInliers = 0

        # For each of the images
        for index, imgWithMatching in enumerate(self.dataBase[method_str]):
            # The RANSAC algorithm is computed to calculate the number of inliers
            try:
                _, mask = cv2.findHomography(imgWithMatching.matchingDatabase,
                                             imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
            except Exception as e:
                LOG.info("Line: %s :: Error: %s", print_frame(), str(e))
                mask = None

            if not mask is None:
                # It is checked, from the mask the number of inliers.
                # If the number of inliers is greater than the minimum number of inliers,
                # and is a maximum (it has more inliers than the previous image)
                # then it is considered to be the image that matches the object stored in the database.
                countNonZero = np.count_nonzero(mask)

                if countNonZero >= minInliers and countNonZero > numInliers:
                    numInliers = countNonZero
                    bestIndex = index
                    bestMask = (mask >= 1).reshape(-1)
        # If an image has been obtained as the best image and therefore must have a minimum number of inlers,
        # then finally the keypoints that are inliers are calculated
        # from the mask obtained in findHomography and returned as the best image.
        LOG.info("Line: %s :: bestIndex: %s, bestMask: %s", print_frame(), bestIndex, bestMask)
        if not bestIndex is None:
            bestImage = self.dataBase[method_str][bestIndex]
            inliersWebCam = bestImage.matchingWebcam[bestMask]
            inliersDataBase = bestImage.matchingDatabase[bestMask]

            return bestImage, inliersWebCam, inliersDataBase
        return None, None, None

    # This function calculates the affinity matrix A, paints a rectangle around
    # Of the detected object and paints in a new window the image of the database
    # Corresponding to the recognized object.
    def calculateAffinityMatrixAndDraw(self, bestImage, inliersDataBase, inliersWebCam, imgout):
        # The affinity matrix A
        A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
        A = np.vstack((A, [0, 0, 1]))

        # Calculate the points of the rectangle occupied by the recognized object
        a = np.array([0, 0, 1], np.float)
        b = np.array([bestImage.shape[1], 0, 1], np.float)
        c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
        d = np.array([0, bestImage.shape[0], 1], np.float)
        centro = np.array([float(bestImage.shape[0]) / 2,
                           float(bestImage.shape[1]) / 2, 1], np.float)

        # Multiply the points of the virtual space, to convert them into Real image points
        a = np.dot(A, a)
        b = np.dot(A, b)
        c = np.dot(A, c)
        d = np.dot(A, d)
        centro = np.dot(A, centro)

        # The points are dehomogenized
        areal = (int(a[0] / a[2]), int(a[1] / b[2]))
        breal = (int(b[0] / b[2]), int(b[1] / b[2]))
        creal = (int(c[0] / c[2]), int(c[1] / c[2]))
        dreal = (int(d[0] / d[2]), int(d[1] / d[2]))
        centroreal = (int(centro[0] / centro[2]), int(centro[1] / centro[2]))

        # The polygon and the file name of the image are painted in the center of the polygon
        points = np.array([areal, breal, creal, dreal], np.int32)
        cv2.polylines(imgout, np.int32([points]), 1, (255, 255, 255), thickness=2)
        utilscv.draw_str(imgout, centroreal, bestImage.nameFile.upper())

        # The detected object is displayed in a separate window
        # self.display_image('ImageDetector', bestImage.imageBinary, resize_max=640, wait_time=1)

    def debug_save_image(self, image_data):
        if gabriel.Debug.SAVE_IMAGES:
            self.log_images_counter += 1
            with open(os.path.join(gabriel.Const.LOG_IMAGES_PATH,
                                   "frame-" + gabriel.util.add_preceding_zeros(self.log_images_counter) + ".jpeg"),
                      "w") as f:
                f.write(image_data)

        if gabriel.Debug.SAVE_VIDEO:
            import cv2
            img_array = np.asarray(bytearray(image_data), dtype=np.int8)
            cv_image = cv2.imdecode(img_array, -1)

            if not self.log_video_writer_created:
                self.log_video_writer_created = True
                self.log_video_writer = cv2.VideoWriter(gabriel.Const.LOG_VIDEO_PATH,
                                                        cv2.VideoWriter_fourcc(*'XVID'), 10,
                                                        (cv_image.shape[1], cv_image.shape[0]))
            self.log_video_writer.write(cv_image)

    def display_image(self, display_name, img, wait_time=-1, is_resize=True, resize_method="max", resize_max=-1,
                      resize_scale=1, save_image=False):
        '''
        Display image at appropriate size. There are two ways to specify the size:
        1. If resize_max is greater than zero, the longer edge (either width or height) of the image is set to this value
        2. If resize_scale is greater than zero, the image is scaled by this factor
        '''
        if is_resize:
            img_shape = img.shape
            height = img_shape[0];
            width = img_shape[1]
            if resize_max > 0:
                if height > width:
                    img_display = cv2.resize(img, (resize_max * width / height, resize_max),
                                             interpolation=cv2.INTER_NEAREST)
                else:
                    img_display = cv2.resize(img, (resize_max, resize_max * height / width),
                                             interpolation=cv2.INTER_NEAREST)
            elif resize_scale > 0:
                img_display = cv2.resize(img, (width * resize_scale, height * resize_scale),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                LOG.info("Unexpected parameter in image display. About to exit...")
        else:
            img_display = img

        if save_image:
            self.debug_save_image(img_display)

        cv2.imshow(display_name, img_display)
        cv2.waitKey(wait_time)

    def recognize_image(self, image_data):
        """
        Ref can be found in
        1. https://stackoverflow.com/a/41122349/2049763
        2. https://stackoverflow.com/q/37716120/2049763

        :param image_data:
        :return: computed image
        """
        kp, desc = self.orb.detectAndCompute(image_data, None)
        try:
            for imageFeature in self.dataBase["ORB"]:
                # Match descriptors.
                matches = self.bf.match(imageFeature.desc, desc)

                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)  # compute the descriptors with ORB

                if len(matches) > MIN_MATCH_COUNT:
                    # Draw first 10 matches.
                    image_data = cv2.drawMatches(imageFeature.imageBinary, imageFeature.kp, image_data, kp,
                                                 matches[:MIN_MATCH_COUNT], None, flags=2)
        except Exception as e:
            LOG.info("%s\n" % str(e))

        # self.display_image('input', image_data, resize_max=640, wait_time=1)
        return image_data

    def detect_image(self, image_data):
        """
        Ref can be found in
        1. https://stackoverflow.com/a/41122349/2049763
        2. https://stackoverflow.com/q/37716120/2049763

        :param image_data:
        :return: computed image
        """
        kp, desc = self.orb.detectAndCompute(image_data, None)

        MAX_MATCH_COUNT = MIN_MATCH_COUNT
        MAX_MATCH = None
        imageFeature = None
        matchesMask = False

        try:
            for imageFeatures in self.dataBase["ORB"]:
                matches = self.bf.match(imageFeatures.desc, desc)
                matches = sorted(matches, key=lambda x: x.distance)  # compute the descriptors with ORB

                if len(matches) > MAX_MATCH_COUNT:
                    MAX_MATCH_COUNT = len(matches)
                    MAX_MATCH = matches
                    imageFeature = imageFeatures

                    if not matchesMask:
                        matchesMask = True

            if matchesMask:
                LOG.info("Match found ! !! !!!")
                image_data = cv2.drawMatches(imageFeature.imageBinary, imageFeature.kp, image_data, kp,
                                             MAX_MATCH[:MIN_MATCH_COUNT],
                                             None, flags=2)
            else:
                LOG.info("No match found")

        except Exception as e:
            LOG.info("%s\n" % str(e))

        # self.display_image('input', image, resize_max=640, wait_time=1)
        return image_data

    def p4_object_recog(self, frame):
        try:
            t1 = time.time()

            image_in = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_out = frame.copy()

            kp, desc = self.detector.detectAndCompute(image_in, None)

            selectedDataBase = self.dataBase[self.method_str]
            if len(selectedDataBase) > 0:
                # We perform the mutual matching
                imgsMatchingMutuos = self.find_matching_mutuos_optimum(self.method_str, desc, kp)

                minInliers = int(cv2.getTrackbarPos('inliers', 'Features'))
                projer = float(cv2.getTrackbarPos('projer', 'Features'))
                LOG.info("Line: %s :: minInliers: %s, projer: %s", print_frame(), minInliers, projer)

                # The best image is calculated based on the number of inliers.
                # The best image is one that has more number of inliers,
                # but always exceeding the minimum that is indicated in the trackbar 'minInliers'
                bestImage, inliersWebCam, inliersDataBase = self.calculate_best_image_by_num_inliers(self.method_str,
                                                                                                     projer, minInliers)

                if not bestImage is None:
                    # If we find a good image, we calculate the affinity matrix
                    # and paint the recognized object on the screen.
                    self.calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, image_out)

            # Get descriptor dimension of each feature:
            dim = -1
            if desc is not None:
                if len(desc) > 0:
                    dim = len(desc[0])
                else:
                    dim = -1
            # We draw features, and write informative text about the image
            # Only the features are dibuban if the slider indicates it
            if (int(cv2.getTrackbarPos('drawKP', 'Features')) > 0):
                cv2.drawKeypoints(image_out, kp, image_out,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            t1 = 1000 * (time.time() - t1)

            utilscv.draw_str(image_out, (20, 20),
                             "Method {0}, {1} features found, desc. dim. = {2} ".
                             format(self.method_str, len(kp), dim))

            utilscv.draw_str(image_out, (20, 40), "Time (ms): {0}".format(str(t1)))

            # Show results and check keys:
            # self.display_image('Features', imgage_out, resize_max=640, wait_time=1)
            return image_out
        except Exception as e:
            LOG.info("* ** p4_object_recog Error: %s\n" % str(e))
            traceback.print_exc()
