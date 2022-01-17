import torch
import numpy as np 
#from tqdm.notebook import tqdm
from validation.validation_model import ValidationModel


class ANNClassifier(ValidationModel):
    class model(torch.nn.Module):
        def __init__(self, inputs=310, hidden=32, outputs=16):
            super().__init__()
            self.inputs = inputs 
            self.hidden = hidden 
            self.outputs = outputs 

            self.ANN = torch.nn.Sequential(torch.nn.Linear(inputs, hidden, bias=True),
                                          torch.nn.Sigmoid(), 
                                          #torch.nn.BatchNorm1d(hidden), 
                                          torch.nn.Linear(hidden, outputs, bias=True))

        def forward(self, x): 
            x = self.ANN(x)
            return x
    
    def __init__(self, n_epochs=10**5, lr=1e-4, hidden_units=32):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_units = hidden_units
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.printing = False
        
    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        else:
            raise TypeError("Invalid type for move_to")
    
    def train_predict(self, train_features, train_labels, test_features):
        # No. in and outputs :
        inputs = train_features.shape[1]
        outputs = len(np.unique(train_labels))

        #Standardize features
        train_mean = train_features.mean(axis = 0)
        train_std = train_features.std(axis = 0)
        
        train_features_standardized = (train_features - train_mean)/train_std
        test_features_standardized  = (test_features  - train_mean)/train_std
        
        #Moving to CUDA if available 
        train_features_standardized = self.move_to(torch.tensor(train_features_standardized), self.DEVICE)
        train_labels = self.move_to(torch.tensor(train_labels - 1), self.DEVICE)
        test_features_standardized = self.move_to(torch.tensor(test_features_standardized), self.DEVICE)


        #Create NN
        #print(np.unique(train_labels))
        model = ANNClassifier.model(inputs=inputs, hidden=self.hidden_units, outputs=outputs)
        if self.DEVICE == "cuda":
            model.cuda() # <- Should only be run if CUDA is available 
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        #Train NN
        for epoch in range(self.n_epochs):
            pred = model(train_features_standardized.float())
            
            train_labels=train_labels.to(torch.int64)#Y?
            #print(pred[0])
            #print(train_labels.dtype)
            
            loss = criterion(pred, train_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Printing the training accuracy if needed to varify that it works 
            if epoch%100 == 0 and self.printing: 
                print("#"*20)
                print(loss) 
                print(sum(np.argmax(pred.cpu().detach().numpy(), axis=1)==train_labels.cpu().detach().numpy()) / 1600)
                train_labels = self.move_to(torch.tensor(train_labels), self.DEVICE)
        
        
        #Get prediction
        test_preds = model(test_features_standardized.float())
        pred_labels = np.argmax(test_preds.cpu().detach().numpy(), axis=1) + 1

        return pred_labels