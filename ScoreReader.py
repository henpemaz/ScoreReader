

import cv2
import numpy as np

import scipy.ndimage.filters as filters

from matplotlib import pyplot as plt

print("reading")
source = cv2.imread('images/score.png')
template = cv2.imread('images/icons/37_vulture.png', cv2.IMREAD_UNCHANGED)
#b,g,r,a = cv2.split(template_rgba)
#template_rgb = cv2.cvtColor(template_rgba , cv2.COLOR_BGRA2BGR)
template_a = template[:,:,3]
template_rgb = template[:,:,:3]
mask = cv2.cvtColor(template_a, cv2.COLOR_GRAY2BGR)
template_bgr = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2BGR)

print("scanning")
result = cv2.matchTemplate(source, template_rgb, cv2.TM_SQDIFF, mask=mask)

result_max_filter = filters.maximum_filter(result, 5)
result_local_maxima = (result == result_max_filter)
result_min_filter = filters.minimum_filter(result, 5)
result_local_minima = (result == result_min_filter)

fig, ax = plt.subplots()
plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
#plt.imshow(cv2.convertScaleAbs(result))

(min_val, max_val, minloc, maxloc) = cv2.minMaxLoc(result)
threshold = min_val*1.1
loc = np.where( (result <= threshold) & result_local_minima)


print("drawing")
i=0
for pt in zip(*loc[::-1]):
    #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    rect = plt.Rectangle(pt, template.shape[1], template.shape[0], edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    i+=1
    if i>100:
        break

plt.show()

