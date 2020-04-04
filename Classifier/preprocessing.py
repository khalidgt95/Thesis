import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

annotations = glob('/home/khalid/Documents/Thesis_work/BCCD_Dataset-master/BCCD/Annotations/*.xml')

df = []
cnt = 0

for file in annotations:
    prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
    filename = prev_filename
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        blood_cells = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)
        
        row = [prev_filename, filename, blood_cells, xmin, xmax,
        ymin, ymax]
        df.append(row)
        cnt += 1

data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type','xmin', 'xmax', 'ymin', 'ymax'])

#data[['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv("train.csv")
from matplotlib import pyplot as plt
from matplotlib import patches
train = pd.read_csv("train.csv")

#plt.imshow(img)

#train["filename"].nunique()

#train["cell_type"].value_counts()

fig = plt.figure()

#add axes to the image
ax = fig.add_axes([0,0,1,1])


image = plt.imread("/home/khalid/Documents/Thesis_work/BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00010.jpg")
plt.imshow(image)
# iterating over the image for different objects
for _,row in train[train.filename == "BloodImage_00010.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    if row.cell_type == 'RBC':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        edgecolor = 'b'
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    ax.add_patch(rect)
    
    
data = pd.DataFrame()
data['format'] = train['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = '/home/khalid/Documents/Thesis_work/BCCD_Dataset-master/BCCD/JPEGImages/' + data['format'][i]

data.iloc[0,:].tolist()

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['cell_type'][i]

data.to_csv('/home/khalid/Documents/Thesis_work/annotate.txt', header=None, index=None, sep=' ')        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    