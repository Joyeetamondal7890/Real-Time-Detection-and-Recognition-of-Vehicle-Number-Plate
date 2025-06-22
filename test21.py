import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils
import random
import openpyxl
import csv
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Detected Plates"
sheet.append(["Number Plate"])
output_file = "./dnp.xlsx"
img = cv2.imread('./Dataset/2.jpeg')  
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.title('Processed Image')
plt.show()
edged = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title('Edge Detection')
plt.show()
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
print("Location: ", location)
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)  
new_image = cv2.bitwise_and(img, img, mask=mask)  
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Masked Image')
plt.show()
(x, y) = np.where(mask == 255)  
(x1, y1) = (np.min(x), np.min(y)) 
(x2, y2) = (np.max(x), np.max(y))  
cropped_image = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.show()
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
text = result[0][-2]
text = text.replace(" ", "")
print("Detected Text: ", text)
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Final Image with Text')
plt.show()
sheet.append([text])
wb.save(output_file)
print(f"\nAll detected plates saved successfully!")

#Quick sort
def partition(arr,low,high): 
    i = ( low-1 )         
    pivot = arr[high]    
  
    for j in range(low , high): 
        if   arr[j] < pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)
        
    return arr
 
#Binary search   
def binarySearch (arr, l, r, x): 
  
    if r >= l: 
        mid = l + (r - l) // 2
        if arr[mid] == x: 
            return mid 
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        else: 
            return binarySearch(arr, mid + 1, r, x) 
    else: 
        return -1
    
def search_csv(filename, text):
    flag=0
    """Searches for text in a CSV file and prints rows containing the text."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader) # Skip the header row

            for row in reader:
                for field in row:
                    if text in field:
                        flag=1
                        break 
            if flag==1:
                print("allowed")
            else:
                print("not allowed")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")

# Example usage
filename = 'num.csv'

search_csv(filename, text)
'''array=[]


with open('num.csv', 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        array.append(row)
    
array=list(array)
if text in array:
     print("allowed")
else:
     print("not allowed")'''
'''#Sorting
array=quickSort(array,0,len(array)-1)
print ("\n\n")
print("The Vehicle numbers registered are:-")
for i in array:
    print(i)
print ("\n\n")    

result = binarySearch(array,0,len(array)-1,[text])
     
    
if result != -1: 
	    print ("\n\nThe Vehicle is allowed to visit.") 
else: 
         print ("\n\nThe Vehicle is  not allowed to visit.")'''
        