import cv2
import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    dr=np.arange(disp_range[0],disp_range[1]+1)
    # create a blank image
    new = np.zeros((img_l.shape[0], img_l.shape[1]))
    # go over all the value in the left image
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            # go over the "window" around r,c
            min= float("inf")
            best_d=-1
            sum = 0
            for i in range(k_size * 2 + 1):
                for j in range(k_size * 2 + 1):
                    # check if the values are in the other image
                    if 0 <= (r + i -(k_size * 2 + 1)//2) < img_r.shape[0] and 0 <= (c + j-(k_size * 2 + 1) // 2) < img_r.shape[1]:
                        # add to sum this number from the equation
                        sum += (img_l[r][c] - img_r[r + i - (k_size * 2 + 1)// 2][c + j -(k_size * 2 + 1) // 2]) ** 2
            # put the sum into the new image
            new[r][c] =sum

    return new


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    # create a blank image
    new = np.zeros((img_l.shape[0], img_l.shape[1]))
    # go over all the value in the left image
    for r in range(img_l.shape[0]):
        for c in range(img_l.shape[1]):
            # go over the "window" around r,c
            top = 0
            bottom1 = 0
            bottom2 = 0
            for i in range(k_size * 2 + 1):
                for j in range(k_size * 2 + 1):
                    # check if the values are in the other image
                    if 0 <= (r + i - disp_range[0] // 2) < img_r.shape[0] and 0 <= (c + j - disp_range[1] // 2) < img_r.shape[1]:
                        # add to top this number from the equation
                        top += (img_l[r][c] * img_r[r + i - disp_range[0] // 2][c + j - disp_range[1] // 2]) ** 2
                        bottom1 += (img_r[r][c] * img_r[r + i - disp_range[0] // 2][c + j - disp_range[1] // 2]) ** 2
                        bottom2 += (img_l[r][c] * img_l[r + i -disp_range[0] // 2][ c + j - disp_range[1] // 2]) ** 2
                    # if 0 <= (disp_range[0]//2+i-disp_range[0]//2) < img_l.shape[0] and 0 <= (disp_range[1]//2+j-disp_range[1]//2) < img_l.shape[1] :
                    #     # and 0 <= disp_range[0] // 2 < img_l.shape[0] and 0 <= disp_range[1] // 2 < img_l.shape[1]
                    #     bottom2 += (img_l[disp_range[0] // 2][disp_range[1] // 2] *img_l[disp_range[0] // 2 + i - disp_range[0] // 2][disp_range[1] // 2 + j - disp_range[1] // 2]) ** 2
                    # if 0 <= (r + i - (k_size * 2 + 1) // 2) < img_r.shape[0] and 0 <= (c + j - (k_size * 2 + 1) // 2) < img_r.shape[1]:
                    #     # add to top this number from the equation
                    #     top += (img_l[r][c] * img_r[r + i - (k_size * 2 + 1)// 2][c + j - (k_size * 2 + 1) // 2]) ** 2
                    #     bottom1 += (img_r[r][c] * img_r[r + i - (k_size * 2 + 1) // 2][c + j - (k_size * 2 + 1)// 2]) ** 2
                    #     bottom2 += (img_l[r][c] * img_l[r+ i -(k_size * 2 + 1) // 2][c+ j -(k_size * 2 + 1) // 2]) ** 2
            # put the sum into the new image
            if (bottom1 * bottom2) != 0:
                new[r][c] = top / np.sqrt(bottom1 * bottom2)
            else:
                new[r][c] = top

    return new


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # create vector A
    A = np.zeros((src_pnt.shape[0] * 2, 9))
    # A=np.array([])
    for x in range(src_pnt.shape[0]):
        A[2*x] = np.array([src_pnt[x][0], src_pnt[x][1], 1, 0, 0, 0, -dst_pnt[x][0]*src_pnt[x][0], -dst_pnt[x][0] * src_pnt[x][1],-dst_pnt[x][0]])
        A[2*x+1] = np.array([0, 0, 0, src_pnt[x][0], src_pnt[x][1], 1, -dst_pnt[x][1] * src_pnt[x][0], -dst_pnt[x][1] * src_pnt[x][1],-dst_pnt[x][1]])


    ATA = A.T @ A
    evalue, evector = np.linalg.eig(ATA)
    max = float("inf")
    spot = 0
    for i in range(len(evalue)):
        if evalue[i] < max:
            max = evalue[i]
            spot = i
    h = evector[spot]

    h = h / h[len(h)-1]
    hT = h.reshape(h.shape[0], 1)
    hT = hT.reshape((3, 3))


    # print(hT)
    u, s, vh= np.linalg.svd(A)
    hom = vh[-1].reshape(3,3)
    hom = hom/hom[2][2]

    # print(hT)
    # print(hom)
    # print(hT-hom)
    # print("\n\n\n")


    # hT = hT.reshape((3, 3))

    diff = src_pnt[0:3,:]-dst_pnt[0:3,:]
    # ddd= hT.dot(diff)
    ddd=hom.dot(diff)
    # error = np.sqrt(sum((hT.dot(src_pnt) - dst_pnt) ** 2))
    error = np.sqrt((np.sum(ddd)**2))
    return hom ,error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)


    ##### Your Code Here ######


    # find 2 smallest x take x with smallest y = top left corner
    # other is bottem left
    # other 2 points smaller y is top right
    # last is bottom right


    # tr= smallest x smallest y
    # tl =biggest x smallest y
    # bl= biggest x smalllest y
    # br smallest x biggest y


    minx1 = float("inf")
    minx2 = float("inf")
    minxrow1 =-1
    minxrow2=-1
    for row in range(len(dst_p)) :
        if dst_p[row][0]<=minx1:
            minx1=dst_p[row][0]
            minxrow1=row
    for row in range(len(dst_p)) :
        if minx1<=dst_p[row][0]<=minx2 and row!=minxrow1:
            minx2=dst_p[row][0]
            minxrow2=row
    if (dst_p[minxrow1][1]<dst_p[minxrow2][1]):
        tl=minxrow1
        bl=minxrow2
    else:
        tl=minxrow2
        bl=minxrow1


    lst=[0,1,2,3]
    lst.remove(tl)
    lst.remove(bl)
    if dst_p[lst[0]][1]> dst_p[lst[1]][1]:
        tr=lst[1]
        br=lst[0]
    else:
        tr=lst[0]
        br=lst[1]

    print("read clockwise\n","topleft",tl, dst_p[tl, :], "topright",tr, dst_p[tr,:],"bottomright",br, dst_p[br,:],"bpttomleft",bl, dst_p[bl, :])

    tl_src=np.array([0,0])
    tr_src=np.array([0, src_img.shape[1]])
    br_src=np.array([src_img.shape[0], src_img.shape[1]])
    bl_src=np.array([src_img.shape[0],0])

    src_p=np.zeros((4,2))
    src_p[tl,:] = tl_src
    src_p[tr,:] = tr_src
    src_p[br,:] = br_src
    src_p[bl,:] = bl_src






    mask=np.zeros((dst_img.shape[0], dst_img.shape[1],3))
    # mask2 = np.zeros((dst_img.shape[0], dst_img.shape[1]))
    for j in range(dst_img.shape[0]):
        for i in range(dst_img.shape[1]):
            if i > dst_p[tl][0] and i> dst_p[bl][0]  and i< dst_p[br][0] and i< dst_p[tr][0] \
                    and  j > dst_p[tl][1] and j > dst_p[tr][1] and j < dst_p[br][1] and j < dst_p[bl][1]:
                mask[j][i][0] = 1
                mask[j][i][1] = 1
                mask[j][i][2] = 1
    plt.imshow(mask)
    plt.show()

    hom, e = computeHomography(src_p, dst_p)
    print(hom)
    hom1 , e1= cv2.findHomography(src_p.astype(float), dst_p.astype(float))
    print(hom1)

    src_out=cv2.warpPerspective(src_img,hom1,(dst_img.shape[1], dst_img.shape[0]))
    plt.imshow(src_out)
    plt.show()

    out = dst_img *(1-mask) + src_out * (mask)
    plt.imshow(out)
    plt.show()
