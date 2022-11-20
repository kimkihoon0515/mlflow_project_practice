import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import mlflow 
import warnings
import random
import matplotlib.pyplot as plt
import copy

def seed_everything(seed):
    torch.manual_seed(seed) # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True # 딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) # numpy를 사용할 경우 고정
    random.seed(seed) # 파이썬 자체 모듈 random 모듈의 시드 고정

seed = 42

seed_everything(seed)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100) # MNIST 데이터셋이 28*28로 총 784개의 픽셀로 이루어져있기 때문에 784를 입력 크기로 넣음.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)

        return x5

download_root = 'MNIST_data/'

train_dataset = datasets.MNIST(root=download_root,
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
                         
test_dataset = datasets.MNIST(root=download_root,
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)    

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Net()
loss_function = nn.CrossEntropyLoss() # 실제 정답과 예측값의 차이를 수치화해주는 함수.
learning_rate = 0.001 # AdamW:0.001, SGD:0.1, Adam:0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # loss_function을 최소화 하는 모델의 파라미터를 찾는 방법.
epochs = 10

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

input_example = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
         0.0706, 0.0706, 0.0706, 0.4941, 0.5333, 0.6863, 0.1020, 0.6510, 1.0000,
         0.9686, 0.4980, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1176, 0.1412, 0.3686, 0.6039,
         0.6667, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.8824, 0.6745, 0.9922,
         0.9490, 0.7647, 0.2510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1922, 0.9333, 0.9922, 0.9922,
         0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9843, 0.3647, 0.3216,
         0.3216, 0.2196, 0.1529, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.8588, 0.9922,
         0.9922, 0.9922, 0.9922, 0.9922, 0.7765, 0.7137, 0.9686, 0.9451, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3137,
         0.6118, 0.4196, 0.9922, 0.9922, 0.8039, 0.0431, 0.0000, 0.1686, 0.6039,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0549, 0.0039, 0.6039, 0.9922, 0.3529, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.5451, 0.9922, 0.7451, 0.0078, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0431, 0.7451, 0.9922, 0.2745,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1373, 0.9451,
         0.8824, 0.6275, 0.4235, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.3176, 0.9412, 0.9922, 0.9922, 0.4667, 0.0980, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.1765, 0.7294, 0.9922, 0.9922, 0.5882, 0.1059, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0627, 0.3647, 0.9882, 0.9922, 0.7333,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9765, 0.9922,
         0.9765, 0.2510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1804, 0.5098, 0.7176, 0.9922,
         0.9922, 0.8118, 0.0078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.1529, 0.5804, 0.8980, 0.9922, 0.9922,
         0.9922, 0.9804, 0.7137, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0941, 0.4471, 0.8667, 0.9922, 0.9922, 0.9922,
         0.9922, 0.7882, 0.3059, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0902, 0.2588, 0.8353, 0.9922, 0.9922, 0.9922, 0.9922,
         0.7765, 0.3176, 0.0078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0706, 0.6706, 0.8588, 0.9922, 0.9922, 0.9922, 0.9922, 0.7647,
         0.3137, 0.0353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.2157, 0.6745, 0.8863, 0.9922, 0.9922, 0.9922, 0.9922, 0.9569, 0.5216,
         0.0431, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.5333, 0.9922, 0.9922, 0.9922, 0.8314, 0.5294, 0.5176, 0.0627,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000]],dtype=np.float32)

input_schema = Schema([
    TensorSpec(np.dtype(np.float32),(-1,784))
])

output_schema = Schema([
    TensorSpec(np.dtype(np.float32),(-1,10))
])

signature = ModelSignature(inputs=input_schema,outputs=output_schema)

model_name = 'basic_model'

warnings.filterwarnings(action='ignore')
experiment_name = '혼돈의 AIP' # 실험명, 실험관리를 용이하게 해줍니다. 


if not mlflow.get_experiment_by_name(experiment_name): 
  mlflow.create_experiment(name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

mlflow.set_tracking_uri('http://127.0.0.1:5001') # 로컬 서버에 실행을 기록하기 위해 함수 호출


best_accuracy = 0
model.zero_grad()

with mlflow.start_run(experiment_id=experiment.experiment_id,run_name="boom") as run:
  for epoch in range(epochs):
    
    model.train() # 학습
    train_accuracy = 0
    train_loss = 0

    for images, labels in train_loader:
      images = images.reshape(batch_size,784)
      image = model(images)
      loss = loss_function(image,labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      prediction = torch.argmax(image,1)
      correct = (prediction == labels)
      train_accuracy+= correct.sum().item() / len(train_dataset)
      train_loss += loss.item() / len(train_loader)

    model.eval() # 평가
    val_accuracy = 0
    val_loss = 0

    for images,labels in test_loader:
      images = images.reshape(batch_size,784)
      image = model(images)
      loss = loss_function(image,labels)
      
      correct = (torch.argmax(image,1) == labels)
      val_accuracy += correct.sum().item() / len(test_dataset)
      val_loss += loss.item() / len(test_loader)
    
    print(f'epoch: {epoch}/{epochs} train_loss: {train_loss:.5} train_accuracy: {train_accuracy:.5} val_loss: {val_loss:.5} val_accuracy: {val_accuracy:.5}')

    if best_accuracy < val_accuracy: # 성능이 가장 좋은 모델로 갱신
      best_accuracy = val_accuracy
      torch.save(model.state_dict(),'best_model.pt')
      best_model = copy.deepcopy(model)
      print(f"===========> Save Model(Epoch: {epoch}, Accuracy: {best_accuracy:.5})")
      
    mlflow.log_param('learning-rate',learning_rate) # mlflow.log_param 을 사용하여 MLflow에 파라미터들을 기록할 수 있습니다.
    mlflow.log_param('epoch',epochs)
    mlflow.log_param('batch_size',batch_size)
    mlflow.log_param('seed',seed)
    mlflow.log_param('optimizer',optimizer)
    mlflow.log_param('loss_function',loss_function)

    mlflow.log_metric('train_accuracy',train_accuracy) # mlflow.log_metric을 사용하여 MLflow에 성능평가를 위한 metric을 기록할 수 있습니다.
    mlflow.log_metric('train_loss',train_loss)
    mlflow.log_metric('valid_accuracy',val_accuracy)
    mlflow.log_metric('valid_loss',val_loss)

    print("--------------------------------------------------------------------------------------------")
  mlflow.pytorch.log_model(best_model,'best_model',signature=signature,input_example=input_example,registered_model_name=model_name) # mlflow.log_model을 사용하여 모델을 mlflow에 저장할 수 있습니다.
print('best model saved to mlflow server')
mlflow.end_run()