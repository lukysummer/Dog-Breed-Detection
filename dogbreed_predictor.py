import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import cv2 
from PIL import Image, ImageFile 
from glob import glob

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()


################### 1. TRANSFORM IMAGE DATA & BUILD LOADERS ###################
transform = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomRotation(10),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

    'valid' : transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

    'test' : transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    }


train_data = datasets.ImageFolder('/data/dog_images/train', 
                                  transform = transform['train'])
valid_data = datasets.ImageFolder('/data/dog_images/valid', 
                                  transform = transform['valid'])
test_data = datasets.ImageFolder('/data/dog_images/test', 
                                 transform = transform['test'])

print('# of Dog Training Images: ', len(train_data))
print('# of Dog Validation Images: ', len(valid_data))
print('# of Dog Test Images: ', len(test_data))
print()

num_workers = 0
batch_size = 20

loaders = {}
loaders['train'] = torch.utils.data.DataLoader(train_data, 
                                               batch_size = batch_size, 
                                               num_workers = num_workers, 
                                               shuffle = True)
loaders['valid'] = torch.utils.data.DataLoader(valid_data, 
                                               batch_size = batch_size, 
                                               num_workers = num_workers, 
                                               shuffle = True)
loaders['test'] = torch.utils.data.DataLoader(test_data, 
                                              batch_size = batch_size, 
                                              num_workers = num_workers, 
                                              shuffle = True)


################## 2. BUILD THE MODEL W/ PRE-TRAINED RESNET ###################
# (1) Import the pre-trained RESNET18 model
resnet_model = models.resnet18(pretrained = True)
model_transfer = resnet_model
if use_cuda:
    model_transfer = model_transfer.cuda()

# (2) Freeze the gradients of the model
for param in model_transfer.parameters():
    param.requires_grad = False

# (3) Change the model's fully connected layer to the desired outputsize of 133 (= number of dogbreeds)
model_transfer.fc = nn.Linear(model_transfer.fc.in_features, 133)    


##################### 3. TRAIN THE MODEL'S LAST FC LAYER ######################
def train(n_epochs, 
          loaders, 
          model,  
          criterion, 
          lr,
          use_cuda, 
          save_model_path):
    
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs + 1):
        optimizer = optim.Adam(model.fc.parameters(), lr = lr[epoch-1])
        train_loss = 0.0
        valid_loss = 0.0

        # (1) TRAIN THE MODEL
        model.train()
        for data, target in loaders['train']:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
                          
        # (2) VALIDATE THE MODEL
        model.eval()
        for data, target in loaders['valid']:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
        
        train_loss = train_loss / len(train_data)
        valid_loss = valid_loss / len(valid_data)
        print('Epoch: {} \t Training Loss: {:.9f} \t Validation Loss: {:.9f}'.format(epoch, train_loss, valid_loss))
        
        if(valid_loss < valid_loss_min):
            torch.save(model.state_dict(), 'model_transfer.pt')
            valid_loss_min = valid_loss
            print('New optimal model!')
            
    return model


n_epochs = 30
criterion = nn.CrossEntropyLoss()
lr = [0.0003]*6 + [0.0001]*8 + [0.00007]*(n_epochs-6-8) # decay learning rate    
start = datetime.now()
model_transfer = train(n_epochs = n_epochs, 
                       loaders = loaders, 
                       model = model_transfer, 
                       criterion = criterion,
                       lr = lr,
                       use_cuda = use_cuda, 
                       save_model_path = 'model_transfer.pt')

print("Entire train runtime: ", datetime.now() - start)

# Load the model that with the lowest validation loss
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


####################### 4. TEST THE MODEL FOR ACCURACY ########################
def test(loaders, model, criterion, use_cuda):
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('\nTest Accuracy: %2d%%' % (100. * correct/total))

test(loaders, model_transfer, criterion, use_cuda)


########################## 5. PREDICT THE DOG BREED ###########################
# list of class names by index, i.e. a name can be accessed like class_names[0]
dog_class_names = [item[4:].replace("_", " ") for item in train_data.classes]

def dogbreed_name(img_path):
    '''Displays the image and returns the predicted dog breed/ most resembling dog breed
    : param img_path: path to the dog/human image
    '''
    # (1) Load & display the image
    img = Image.open(img_path).convert('RGB') 
    plt.imshow(img)
    plt.show()    
        
    # (2) Apply transform to the image
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = transform(img)[:3, :, :].unsqueeze(0)
    
    # (3) Feed the image into the trained model & predict the breed
    model_transfer.eval()
    if use_cuda:
        img = img.cuda()
    output = model_transfer(img).cpu().data.numpy().argmax()
    
    return dog_class_names[output]


def human_face_detector(img_path):
    '''Returns True (1) if a human face is deteced in img_path, if not returns False (0)'''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    
    return len(faces) > 0


def dog_detector(img_path):  
    '''
    Use a pre-trained model to obtain index corresponding to predicted ImageNet class
    : param img_path: path to image
    : Returns:   Index corresponding to VGG-16 model's prediction class
    '''
    img = Image.open(img_path).convert('RGB') 
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    img = transform(img)[:3, :, :].unsqueeze(0)
    
    
    if use_cuda:
        img = img.cuda()
    class_index = resnet_model(img).cpu().data.numpy().argmax()
    
    return (class_index >= 151 and class_index <= 268)


def predict_dogbreed(img_path):
    vowels = "AIEOUY"
    
    if human_face_detector(img_path) == 1:
        dog = dogbreed_name(img_path)
        if dog[0] not in vowels:
            print("You look like a {}.".format(dog), '\n')
        else:
            print("You look like an {}.".format(dog), '\n')
    
    elif dog_detector(img_path, resnet_model) == 1:
        dog = dogbreed_name(img_path)
        if dog[0] not in vowels:
            print("This is a {}.".format(dog), '\n')
        else:
            print("This is an {}.".format(dog), '\n')
        
    else: 
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        print('EROOR: Please input an image of a human or a dog!\n')
        
# Now, make prediction!        
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))
        
for file in np.hstack((human_files[:5], dog_files[:5])):
    predict_dogbreed(file)