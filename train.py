import argparse
import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import mlflow # mlflow 사용을 위해
import torch.backends.cudnn as cudnn
import random
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100)
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
                         transform = transforms.ToTensor(),
                         download=True)
                         
test_dataset = datasets.MNIST(root=download_root,
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)    

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)

def train(args):

  model = Net()
  model.zero_grad()
  loss_function = nn.CrossEntropyLoss()
  learning_rate = args.lr
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

  experiment_name = 'chaos_AIP' # 실험명, 실험관리를 용이하게 해줍니다. 


  if not mlflow.get_experiment_by_name(experiment_name): 
    mlflow.create_experiment(name=experiment_name)
  experiment = mlflow.get_experiment_by_name(experiment_name)

  mlflow.set_tracking_uri('http://127.0.0.1:5000') # 로컬 서버에 실행을 기록하기 위해 함수 호출
  mlflow.set_experiment(experiment_name) # 실험 
  #mlflow.set_tag("mlflow.runName","practice")

  train_loss_list = []
  train_acc_list = []

  val_loss_list = []
  val_acc_list = []

  total_batch = len(train_loader)

  epochs = args.epochs

  best_accuracy = 0
  with mlflow.start_run(run_name="boom"):
    for epoch in range(epochs):
        cost=0
        model.train()
        train_accuracy = 0
        train_loss = 0
        for images, labels in train_loader:
            images = images.reshape(100,784)
            
            optimizer.zero_grad() # 변화도 매개변수 0
            
            #forward
            #pred = model.forward(images)
            #loss = loss_function(pred, labels)
            pred = model(images)
            loss = loss_function(pred,labels)
            prediction = torch.argmax(pred,1)
            correct = (prediction == labels)
            train_accuracy += correct.sum().item() / 60000
            train_loss += loss.item() / 600
            
            #backward
            loss.backward()
            
            #Update
            optimizer.step()
            
            cost += loss
        
        with torch.no_grad(): #미분하지 않겠다는 것
            total = 0
            correct=0
            for images, labels in test_loader:
                images = images.reshape(100,784)

                outputs = model(images)
                _,predict = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predict==labels).sum() # 예측한 값과 일치한 값의 합

        avg_cost = cost / total_batch
        accuracy = 100*correct/total

        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(avg_cost.detach().numpy())
        val_acc_list.append(accuracy)

        if accuracy > best_accuracy:
          torch.save(model.state_dict(),'model.pt')
          best_accuracy = accuracy
          print(f"Save Model(Epoch: {epoch+1}, Accuracy: {best_accuracy:.5})")
        
        print("epoch : {} | loss : {:.6f}" .format(epoch+1, avg_cost))
        print("Accuracy : {:.2f}".format(100*correct/total))
        mlflow.log_param('learning-rate',learning_rate) # mlflow.log_param 을 사용하여 MLflow에 파라미터들을 기록할 수 있습니다.
        mlflow.log_param('epoch',epochs)
        mlflow.log_param('batch_size',batch_size)
        mlflow.log_param('seed',seed)
        mlflow.log_metric('train_accuracy',train_accuracy) # mlflow.log_metric을 사용하여 MLflow에 성능평가를 위한 metric을 기록할 수 있습니다.
        mlflow.log_metric('train_loss',train_loss)
        mlflow.log_metric('valid_accuracy',accuracy)
        mlflow.log_metric('valid_loss',avg_cost)
        mlflow.pytorch.log_model(model,'model') # pytorch.log_model 을 통해 모델을 저장할 수 있습니다.
        print("------")
  mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr',type=float,default=0.03,help='learning rate')
    parser.add_argument('--epochs',type=int,default=10,help='epoch limits')

    args = parser.parse_args()
    print(args)
    train(args)