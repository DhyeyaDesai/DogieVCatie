import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import torch.optim as optim
from PIL import ImageTk, Image


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*2*2, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        x = x.flatten(start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

def predict(event):
	try:
		location = imageLocation.get()

		openImage(location)

		img = cv2.imread(location, 0)
		img = cv2.resize(img, (50, 50))
		img = np.array(img)
		img = torch.tensor(img).view(-1, 1, 50, 50)
		img = img/255.0
		op = net(img)

		if torch.argmax(op).item() == 0:
			prediction = "I'm " + str((op.max().item())*100) + " % sure It's a cat.\nAm I right?"
			prediction_vector = torch.tensor([1., 0.])

		else:
			prediction = "I'm " + str((op.max().item())*100) + " % sure It's a dog.\nAm I right?"
			prediction_vector = torch.tensor([0., 1.])

		feedback = messagebox.askquestion("This is my Prediction", prediction)
		trainModel(img, op.squeeze(), prediction_vector)
		

	except Exception as e:
		print(str(e))

def trainModel(input, op, label):
    # print(op)

    optimizer.zero_grad()
    loss = loss_function(op, label)
    loss.backward()
    optimizer.step()

    torch.save(net, "DogieVCatie.pt")

def openImage(location):
	pet_image = ImageTk.PhotoImage(Image.open(location))
	image_label = Label(image = pet_image)
	image_label.grid(row = 3, column = 0)


net = torch.load("DogieVCatie.pt")

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

root = Tk()
root.geometry("600x500")

root.title("DogieVCatie")


label = Label(root, text = "Enter the location of an image: ").grid(row = 0, column = 0)

imageLocation = Entry(root, width = 50)
imageLocation.grid(row = 0, column = 1)

predictButton = Button(root, text = "Predict")
predictButton.grid(row = 0, column = 8)
predictButton.bind("<Button-1>", predict)

root.mainloop()