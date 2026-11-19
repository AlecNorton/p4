import cv2
import numpy as np

def furthest(mask, pixel_threshold = 50):        
        cv2.imwrite('mask.png', mask)
        img = np.asarray(mask).astype('uint8')
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img.shape
        empty = np.zeros((height, width, 1), np.uint8)
        returnPt = None
        maxPixels = 0
        count = 0
        for c in contours:
            one_img = cv2.drawContours(empty.copy(), [c], 0, (255, 255, 255), -1)
            mask_img = (one_img.copy() / 255).astype(int)

            one_img = cv2.cvtColor(one_img, cv2.COLOR_GRAY2BGR)
            num = np.count_nonzero(mask_img)
            if(num <= pixel_threshold):
                #Not a mask
                print("too small; not a mask, continuing.")
                continue
            mask_img = np.squeeze(mask_img)
            indices=  np.nonzero(mask_img)
            avg = np.mean(indices, axis = 1).astype('int')
            raw_dist = np.zeros(mask_img.shape, dtype=np.float32)
            raw_dist = np.pad(raw_dist, 1, 'constant')
            for pt_num in range(0, len(indices[0])):
                x, y = indices[0][pt_num], indices[1][pt_num]
                pt = np.float32((y, x))
                raw_dist[x,y] = cv2.pointPolygonTest(c, pt, True)
            minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
            # Depicting the  distances graphically
            cv2.circle(one_img, maxDistPt, 5,(255,0,0), -1)
            cv2.circle(one_img, (avg[1], avg[0]), 5, (0, 255, 0), -1)
            cv2.imwrite(f"mask_{count}.png", one_img)
            if(maxPixels < num):
                 returnPt = maxDistPt
                 maxPixels = num
                 biggest_gap = one_img.copy()
            count += 1
        cv2.imwrite("final_mask.png", biggest_gap)
            
        
        return returnPt
