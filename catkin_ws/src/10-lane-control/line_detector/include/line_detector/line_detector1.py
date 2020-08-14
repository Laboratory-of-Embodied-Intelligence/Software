import cv2
import duckietown_utils as dtu

import numpy as np

from .line_detector_interface import Detections, LineDetectorInterface


class LineDetectorHSV(dtu.Configurable, LineDetectorInterface):
    """ LineDetectorHSV """

    def __init__(self, configuration):
        # Images to be processed
        self.bgr = np.empty(0)
        self.hsv = np.empty(0)
        self.edges = np.empty(0)

        param_names = [
            'hsv_white1',
            'hsv_white2',
            'hsv_yellow1',
            'hsv_yellow2',
            'hsv_red1',
            'hsv_red2',
            'hsv_red3',
            'hsv_red4',
            'dilation_kernel_size',
            'canny_thresholds',
            'hough_threshold',
            'hough_min_line_length',
            'hough_max_line_gap',
        ]

        dtu.Configurable.__init__(self, param_names, configuration)

    def _colorFilter(self, color):
        # threshold colors in HSV space
        if color == 'white':
            bw = cv2.inRange(self.hsv, self.hsv_white1, self.hsv_white2)
        elif color == 'yellow':
            bw = cv2.inRange(self.hsv, self.hsv_yellow1, self.hsv_yellow2)
        elif color == 'red':
            bw1 = cv2.inRange(self.hsv, self.hsv_red1, self.hsv_red2)
            bw2 = cv2.inRange(self.hsv, self.hsv_red3, self.hsv_red4)
            bw = cv2.bitwise_or(bw1, bw2)
        else:
            raise Exception('Error: Undefined color strings...')

        # binary dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.dilation_kernel_size, self.dilation_kernel_size))
        bw = cv2.dilate(bw, kernel)

        # refine edge for certain color
        edge_color = cv2.bitwise_and(bw, self.edges)

        return bw, edge_color

    def hist_binarize(self, img, k ):
        '''
        Returns the binarized mask based on information from L, a and b channels of the image in CIELAB.
        The mask from L channel (bin_res_l) corresponds to light objects on the image (all road markings)
        The mask from a channel (bin_res_a) corresponds to red objects on the image (red road markings)
        The mask from b channel (bin_res_b) corresponds to yellow on the image (yellow road markings)

        Parameters:
            img (np.array): Image of the road in RGB format (only ROI).
        k (double): scaling coefficient, depends on the size of the road marking (for DT k~=1, for real roads k~=2)
        
        Returns:
            bin_res_l, bin_res_a, bin_res_b(np.array): Binarized images of L, a, and b channels.   
        '''
        img = cv2.GaussianBlur(img,(5,5),0)
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split( img )
        
        norm_l = cv2.normalize(l, None, alpha=0 , beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_a = cv2.normalize(a, None, alpha=0 , beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_b = cv2.normalize(b, None, alpha=0 , beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
        mu_l, sigma_l = cv2.meanStdDev( norm_l )
        mu_l = mu_l[0]
        sigma_l = sigma_l[0]
        t_l = int( mu_l + sigma_l * (k + sigma_l * np.sqrt(3)/(255)) )
        _, thresh_l = cv2.threshold( norm_l, t_l, 1, cv2.THRESH_BINARY )

        mu_a, sigma_a = cv2.meanStdDev( norm_a )
        mu_a = mu_a[0]
        sigma_a = sigma_a[0]
        t_a = int( mu_a + sigma_a * (k + sigma_a * np.sqrt(3)/(255)) )
        _, thresh_a = cv2.threshold( norm_a, t_a, 1, cv2.THRESH_BINARY )
            
        mu_b, sigma_b = cv2.meanStdDev( norm_b )
        mu_b = mu_b[0]
        sigma_b = sigma_b[0]
        t_b = int( mu_b + sigma_b * (k + sigma_b * np.sqrt(3)/(255)) )
        _, thresh_b = cv2.threshold( norm_b, t_b, 1, cv2.THRESH_BINARY )

        kernel = np.ones((3, 3))
        bin_res_l = cv2.morphologyEx(thresh_l, cv2.MORPH_CLOSE, kernel)
        bin_res_a = cv2.morphologyEx(thresh_a, cv2.MORPH_CLOSE, kernel)
        bin_res_b = cv2.morphologyEx(thresh_b, cv2.MORPH_CLOSE, kernel)
            
        return bin_res_l, bin_res_a, bin_res_b

    def _colorFilter_binarize(self, color):
        # threshold colors in HSV space

        if color == 'white':
            bw = self.white
        elif color == 'yellow':
            bw = self.yellow
        elif color == 'red':
            bw = self.red
        else:
            raise Exception('Error: Undefined color strings...')

        # binary dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.dilation_kernel_size, self.dilation_kernel_size))
        bw = cv2.dilate(bw, kernel)

        # refine edge for certain color
        edge_color = cv2.bitwise_and(bw, self.edges)

        return bw, edge_color


    def _findEdge(self, gray):
        edges = cv2.Canny(gray, self.canny_thresholds[0], self.canny_thresholds[1], apertureSize=3)
        return edges

    def _HoughLine(self, edge):
        lines = cv2.HoughLinesP(edge, 1, np.pi / 180, self.hough_threshold, np.empty(1),
                                self.hough_min_line_length, self.hough_max_line_gap)
        if lines is not None:
            lines = np.array(lines[:, 0])
        else:
            lines = []
        return lines

    def _checkBounds(self, val, bound):
        val[val < 0] = 0
        val[val >= bound] = bound - 1
        return val

    def _correctPixelOrdering(self, lines, normals):
        flag = ((lines[:, 2] - lines[:, 0]) * normals[:, 1] - (lines[:, 3] - lines[:, 1]) * normals[:, 0]) > 0
        for i in range(len(lines)):
            if flag[i]:
                x1, y1, x2, y2 = lines[i, :]
                lines[i, :] = [x2, y2, x1, y1]

    def _findNormal(self, bw, lines):
        normals = []
        centers = []
        if len(lines) > 0:
            length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
            dx = 1.* (lines[:, 3:4] - lines[:, 1:2]) / length
            dy = 1.* (lines[:, 0:1] - lines[:, 2:3]) / length

            centers = np.hstack([(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
            x3 = (centers[:, 0:1] - 3.*dx).astype('int')
            y3 = (centers[:, 1:2] - 3.*dy).astype('int')
            x4 = (centers[:, 0:1] + 3.*dx).astype('int')
            y4 = (centers[:, 1:2] + 3.*dy).astype('int')
            x3 = self._checkBounds(x3, bw.shape[1])
            y3 = self._checkBounds(y3, bw.shape[0])
            x4 = self._checkBounds(x4, bw.shape[1])
            y4 = self._checkBounds(y4, bw.shape[0])
            flag_signs = (np.logical_and(bw[y3, x3] > 0, bw[y4, x4] == 0)).astype('int') * 2 - 1
            normals = np.hstack([dx, dy]) * flag_signs

            """ # Old code with lists and loop, performs 4x slower
            for cnt,line in enumerate(lines):
                x1,y1,x2,y2 = line
                dx = 1.*(y2-y1)/((x1-x2)**2+(y1-y2)**2)**0.5
                dy = 1.*(x1-x2)/((x1-x2)**2+(y1-y2)**2)**0.5
                x3 = int((x1+x2)/2. - 3.*dx)
                y3 = int((y1+y2)/2. - 3.*dy)
                x4 = int((x1+x2)/2. + 3.*dx)
                y4 = int((y1+y2)/2. + 3.*dy)
                x3 = self._checkBounds(x3, bw.shape[1])
                y3 = self._checkBounds(y3, bw.shape[0])
                x4 = self._checkBounds(x4, bw.shape[1])
                y4 = self._checkBounds(y4, bw.shape[0])
                if bw[y3,x3]>0 and bw[y4,x4]==0:
                    normals[cnt,:] = [dx, dy]
                else:
                    normals[cnt,:] = [-dx, -dy]
            """
            self._correctPixelOrdering(lines, normals)
        return centers, normals

    def detectLines(self, color):
        with dtu.timeit_clock('_colorFilter'):
            # bw, edge_color = self._colorFilter(color)
            bw, edge_color = self._colorFilter_binarize(color)
        with dtu.timeit_clock('_HoughLine'):
            lines = self._HoughLine(edge_color)
        with dtu.timeit_clock('_findNormal'):
            centers, normals = self._findNormal(bw, lines)
        return Detections(lines=lines, normals=normals, area=bw, centers=centers)

    def setImage(self, bgr):
        with dtu.timeit_clock('np.copy'):
            self.bgr = np.copy(bgr)
        with dtu.timeit_clock('cvtColor COLOR_BGR2HSV'):
            self.hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        with dtu.timeit_clock('_findEdge'):
            self.edges = self._findEdge(self.bgr)

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        koefficient = 2
        bin_res_l, bin_res_a, bin_res_b = self.hist_binarize(image_rgb, koefficient)
        self.yellow = np.uint8(bin_res_b * 255)
        self.white = np.uint8(bin_res_l * 255)
        self.red = np.uint8(bin_res_a * 255)

    def getImage(self):
        return self.bgr
