from pickle import TRUE
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn

# cols=["Evaporation","Precipitation","Temperature","Discharge"]
# df = pd.read_excel('sheet 1.xlsx',usecols=cols)
# # print(df.head(3))


# #Split into parameters and target (Discharge)
# X = df.drop('Discharge', axis = 1)
# y = df['Discharge']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 20)
# # print(X_train)

# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaler.fit(X_train)

# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)


class RiverDataset(Dataset):
    def __init__(self) -> None:
        super(RiverDataset,self).__init__()
        self.cols=["Evaporation","Precipitation","Temperature","Discharge"]
        self.dataframe = pd.read_excel('C:/Users/thtuh/Desktop/project/Final Data/CDC/for python/sheet 1.xlsx',usecols=self.cols)
        self.x = torch.from_numpy(self.dataframe.drop('Discharge', axis = 1).to_numpy(dtype=np.float32))
        self.y = torch.from_numpy(self.dataframe["Discharge"].to_numpy(dtype=np.float32))

    def __getitem__(self, index):
        sample =self.x[index],self.y[index]
        return sample
 
    def __len__(self):
        return len(self.dataframe)

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss,self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class DNN(nn.Module):
    def __init__(self,input_dim,output_dim ) -> None:
        super(DNN,self).__init__()
        self.dnn = nn.Sequential(
                                nn.Linear(input_dim,128),
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.ReLU(),
                                # nn.Linear(64,32),
                                # nn.ReLU(),
                                nn.Linear(64,output_dim))

    def forward(self,x):
        return self.dnn(x)


def train():
    numEpochs =100
    data = RiverDataset()
    train_loader = DataLoader(data,batch_size=200,shuffle=True)
    model = DNN(3,1)
    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)


    # for idx,(x,y) in enumerate(train_loader):
    #         y= y.view(-1,1)
    #         prediction = model(x)
    #         print(y.shape,prediction.shape)
    #         loss = criterion(prediction,y)
    #         print(loss)

    for epoch in range(numEpochs):
        for idx,(x,y) in enumerate(train_loader):
            y= y.view(-1,1)
            prediction = model(x)
            # print(y.shape,prediction.shape)
            loss = criterion(prediction,y)
            loss.backward()
            print(loss)
            optimizer.step()
            optimizer.zero_grad()
            if (idx+1)==10:
                print(f"epoch:{epoch}/{numEpochs}, loss={loss}")


if __name__=="__main__":
    train()

    