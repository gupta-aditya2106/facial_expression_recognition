#importing required libraries
#Basic Imports
import numpy as np 
import matplotlib.pyplot as plt
import torch 

#
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from torch.utils.data import DataLoader

import timm
from torch import nn

from tqdm import tqdm


#Configurations
#Remote control of the program - the constant values to be used are defined here
TRAIN_IMG_FOLDER_PATH = 'C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Facial Expression Recognition/Facial-Expression-Dataset/train/'
VALID_IMG_FOLDER_PATH = 'C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Facial Expression Recognition/Facial-Expression-Dataset/validation/'

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 15

DEVICE = 'cpu'
MODEL_NAME = 'efficientnet_b0'



#Load Dataset
train_augs = T.Compose([
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor() #PIL / numpy arr -> torch tensor -> (h, w, c) -> {c, h, w)
])

valid_augs = T.Compose([
    T.ToTensor()
])

trainset = ImageFolder(TRAIN_IMG_FOLDER_PATH, transform = train_augs)
validset = ImageFolder(VALID_IMG_FOLDER_PATH, transform = valid_augs)

print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")

print(trainset.class_to_idx)

image, label = validset[4000]

plt.imshow(image.permute(1, 2, 0)) #(h, w, c)
plt.title(label)
plt.show()



#Load Dataset into Batches
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)

print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")

for images, labels in trainloader:
  break;

print(f"One image batch shape : {images.shape}") #(No. of images in batch, channel, height, width)
print(f"One label batch shape : {labels.shape}")



#Create Model
class FaceModel(nn.Module):

  def __init__(self):
    super(FaceModel, self).__init__()
    self.eff_net = timm.create_model('efficientnet_b0', pretrained = True, num_classes = 7)

  def forward(self, images, labels = None):
    logits = self.eff_net(images)

    if labels != None:
      loss = nn.CrossEntropyLoss()(logits, labels)
      return logits, loss
    
    return logits

model = FaceModel()
model.to(DEVICE);



#Create Train and Eval Function
def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def train_fn(model, dataloader, optimizer, current_epo):
  model.train()
  total_loss = 0
  total_acc = 0
  tk = tqdm(dataloader, desc = "EPOCH" + "[TRAIN]" + str(current_epo + 1) + "/" + str(EPOCHS))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)
    tk.set_postfix({'loss' : '%6f' %float(total_loss / (t+1)), 'acc' : '%6f' %float(total_acc / (t+1))});

  return total_loss / len(dataloader) , total_acc / len(dataloader)


def eval_fn(model, dataloader, current_epo):
  model.eval()
  total_loss = 0
  total_acc = 0
  tk = tqdm(dataloader, desc = "EPOCH" + "[VALID]" + str(current_epo + 1) + "/" + str(EPOCHS))


  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    logits, loss = model(images, labels)

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)
    tk.set_postfix({'loss' : '%6f' %float(total_loss / (t+1)), 'acc' : '%6f' %float(total_acc / (t+1)),})

  return total_loss / len(dataloader) , total_acc / len(dataloader)



#Create Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

best_valid_loss = np.Inf

for i in range(EPOCHS):
  train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
  valid_loss, valid_acc = eval_fn(model, validloader, i)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best-weights.pt')
    print("SAVED-BEST-WEIGHTS")
    best_valid_loss = valid_loss



#Inference
def view_classify(img, ps):

    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)

    fig, (ax1, ax2) = plt.subplots(figsize=(5,9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None

# Load the best model weights
model.load_state_dict(torch.load('best-weights.pt'))

# Use the model to predict the probabilities for a single image from the validation set
image, label = validset[1005]
image = image.unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    logits = model(image.to(DEVICE))
    probs = nn.Softmax(dim=1)(logits)

# Display the image and its predicted probabilities
view_classify(image.squeeze(), probs)
