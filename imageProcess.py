import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL
from PIL import Image, ImageTk
from tkinter import * 
from tkinter import filedialog 
import imagehash
from cassandra.cluster import Cluster


def browseFiles(): 
        global filename
        filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select an Image", 
                                          filetypes = (("Image files", 
                                                        "*.*"), 
                                                       ("all files", 
                                                        "*.*"))) 
                                                        

        #label_file_explorer.configure(text="Image Opened: "+filename)  
        #window = Tk()  
        #canvas = Canvas(window, width = 300, height = 300)  
        #canvas.pack()  
        #img = ImageTk.PhotoImage(PIL.Image.open(filename))  
        #canvas.create_image(20, 20, anchor=NW, image=img) 
        #window.mainloop()

    
    
window = Tk() 
   
     
window.title('Text Recognition') 
   
     
window.geometry("1240x480") 
   
    
window.config(background = "white") 
   
     
label_file_explorer = Label(window,  
                            text = "Select an Image to recognize text from : ", 
                            width = 200, height = 4,  
                            fg = "blue") 
    
   

def mainfun():
        global content
        content=""
        args = {"image":filename, "east":"./east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}
        args['image']=filename
        print(filename)
        image = cv2.imread(filename)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        orig = image.copy()
        (origH, origW) = image.shape[:2]

        (newW, newH) = (args["width"], args["height"])

        rW = origW / float(newW)
        rH = origH / float(newH)
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net = cv2.dnn.readNet(args["east"])
        
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        
        def predictions(prob_score, geo):
            (numR, numC) = prob_score.shape[2:4]
            boxes = []
            confidence_val = []
            
            for y in range(0, numR):
                scoresData = prob_score[0, 0, y]
                x0 = geo[0, 0, y]
                x1 = geo[0, 1, y]
                x2 = geo[0, 2, y]
                x3 = geo[0, 3, y]
                anglesData = geo[0, 4, y]

            
                for i in range(0, numC):
                    if scoresData[i] < args["min_confidence"]:
                        continue
                    (offX, offY) = (i * 4.0, y * 4.0)
                    angle = anglesData[i]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    h = x0[i] + x2[i]
                    w = x1[i] + x3[i]
                    endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                    endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                    startX = int(endX - w)
                    startY = int(endY - h)
                    boxes.append((startX, startY, endX, endY))
                    confidence_val.append(scoresData[i])
            return (boxes, confidence_val)
            
            
            
            
        (boxes, confidence_val) = predictions(scores, geometry)
        #print(confidence_val)
        boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
        results = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            r = orig[startY:endY, startX:endX]
            configuration = ("-l eng --oem 1 --psm 8")
            text = pytesseract.image_to_string(r, config=configuration)
            results.append(((startX, startY, endX, endY), text))
    
        orig_image = orig.copy()

        for ((start_X, start_Y, end_X, end_Y), text) in results:
            content=content+text
            #print(text)
            #print("{}\n".format(text)) 
            text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
            cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
            (0, 0, 255), 2)
            cv2.putText(orig_image, text, (start_X, start_Y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)
    
        
        window.lift()
        plt.imshow(orig_image)
        plt.title('Output')
        plt.show()

def write():
    hash = imagehash.average_hash(PIL.Image.open(filename))
    hashs=str(hash);
    clstr=Cluster()
    session=clstr.connect('mykeyspace')
    stmt=session.prepare("INSERT INTO images (hash,content) VALUES (?,?)")
    qry=stmt.bind([hashs,content])
    session.execute(qry)
    
def check():
    global count1
    count1=0
    hash=imagehash.average_hash(PIL.Image.open(filename))
    hashs=str(hash)
    clstr=Cluster()
    session=clstr.connect('mykeyspace')
    stmt=session.prepare("SELECT count(hash) as count FROM images WHERE hash= '{}' ".format(hashs))
    count=session.execute(stmt)
    row=count.one()
    print(row.count)
    count1=row.count
    if row.count==1 : 
        print("image already exists")
        clstr.shutdown()
        label_file_explorer.configure(text=""+filename+" already exists", fg= "red")
                             
 
button_exit = Button(master=window,
                     text = "Exit", 
                     command = exit)
                     

button_extract= Button(master=window,
                        text= "Extract",
                        command=mainfun)
                        

button_save= Button(master=window,
                      text="Save",
                      command=write)

def merge():
    browseFiles()
    check()
    

button_explore = Button(window,  
                        text = "Select Image", 
                        command = merge) 

label_file_explorer.grid(column = 1, row = 1) 
   
button_explore.grid(column = 1, row = 2) 
    
button_exit.grid(column = 1,row = 5) 

button_extract.grid(column= 1,row=3)

button_save.grid(column=1,row=4)
   
window.mainloop()