from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
import numpy as np

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

def plot_ellipses(ellipse1, ellipse2, SAVE_PATH, IoU):

    fig,ax = plt.subplots()

    ax.set_xlim([-15,15])
    ax.set_ylim([-5,5])
    ax.set_aspect('equal')

    ax.add_patch(ellipse1)
    
    ax.add_patch(ellipse2)
    
    plt.title("IoU : " + str(IoU))    
    plt.show()
    
    name = SAVE_PATH+str(i)+".png"
    print(name)
    plt.savefig(name)
        
    return

def print_statistics(ellipse1, ellipse2):

    print('area of ellipse 1:',ellipse1.area)
    print('area of ellipse 2:',ellipse2.area)

#    print('intersect ratio for ellipse1:', intersect.area/ellipse1.area)
#    print('intersect ratio for ellipse2:', intersect.area/ellipse2.area)
    
    return

def IoU(ellipse1, ellipse2):

    intersection = ellipse1.intersection(ellipse2)
    
    denominator = ellipse1.area + ellipse2.area - intersection.area
    
    iou = intersection.area / denominator
    
    print('area of intersection:',intersection.area)    
    
    return iou

if __name__=="__main__":

    
    ##first ellipse in blue
    SAVE_PATH = "C:\\Users\\Khalid\\Documents\\4th Semster Thesis\\Thesis work\\Thesis\\Images\\"
    centers = np.arange(-10, 10, 1)
    
    for i in range(len(centers)):
        ellipse1 = create_ellipse((0,0),(2,4),90)
        verts1 = np.array(ellipse1.exterior.coords.xy)
        patch1 = Polygon(verts1.T, color = 'blue', alpha = 0.5)
        
        ##second ellipse in red    
        ellipse2 = create_ellipse((centers[i],0),(2,4),90)
        verts2 = np.array(ellipse2.exterior.coords.xy)
        patch2 = Polygon(verts2.T,color = 'red', alpha = 0.5)
        
        ##the intersect will be outlined in black
        #intersect = ellipse1.intersection(ellipse2)
        #verts3 = np.array(intersect.exterior.coords.xy)
        #patch3 = Polygon(verts3.T, facecolor = 'none', edgecolor = 'black')
        #ax.add_patch(patch3)
        
        IoU_value = IoU(ellipse1, ellipse2)
        plot_ellipses(patch1, patch2, SAVE_PATH, IoU_value)
        ##compute areas and ratios 
        
        print("IoU : {} ".format(IoU_value))
        
        #plt.show()
       