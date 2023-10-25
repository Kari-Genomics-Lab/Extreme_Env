import sys
import torch
import random
import numpy as np



import torch.nn as nn

import torch.optim as optim

import math

from torch.utils.data import DataLoader, Dataset

sys.path.append('../src/')

from LossFunctions import info_nce_loss
from utils import create_dataloader, SummaryFasta, SequenceDataset, kmersFasta


# Random Seeds for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


#from torch.utils.tensorboard import SummaryWriter
global dtype
global EPS

dtype = torch.FloatTensor
long_dtype = torch.LongTensor
EPS = sys.float_info.epsilon

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor


def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)



class SuperNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(SuperNet, self).__init__()
        self.layers = nn.Sequential(
                nn.BatchNorm1d(n_input,eps=1e-14),
                nn.Linear(n_input, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 64),
                nn.LeakyReLU()
        )
        
        self.instance = nn.Linear(64, 64)  # Always check n_input here.
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(64, n_output)  # Always check n_input here.
        )
            
    def forward(self, x):
        x = x.view(-1, x.shape[2])
        x = self.layers(x)
        latent = self.instance(x)
        out = self.classifier(x)
        return out, latent
    
class myDataset(Dataset):
    
    def __init__(self, data, labels=None, transform=None):
        
        if transform: 
            self.data=transform(data)           
        else:
            self.data = data
            
        self.labels = labels

        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return {'features': self.data[idx, :], 'labels': self.labels[idx]}

class supervised_model():

    def __init__(self,  n_clusters=3, batch_sz=512, k=6, epochs=100):
        
        if k % 2 == 0: n_in = (4**k + 4**(k//2))//2 
        else: n_in = (4**k)//2
        #d = {3:25, 4:103, 5:391, 6:1567}
        #n_in = d[k]
        #n_in = 4**k
        self.net = SuperNet(n_in, n_clusters) #NetLinear(2079, n_clusters)
        self.net.apply(weights_init)
        self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters())
        self.n_clusters = n_clusters
        self.batch_sz = batch_sz
        self.epochs = epochs
        
        #self.writer = SummaryWriter()
    
        #print(self.net)
        #print("Number of Trainable Parameters: ", 
        #      sum(p.numel() for p in self.net.parameters() if p.requires_grad))


    def build_dataloaders(self, kmers, GT):
        dataset = myDataset(kmers, GT)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_sz, shuffle=True, num_workers=2)
        
        
    def fit(self, kmers, GT, epochs=100):
        self.build_dataloaders(kmers, GT)
        n_features = kmers.shape[1]
        criterion = nn.CrossEntropyLoss()
        self.net.train()
        
        for e in range(self.epochs):
            for i_batch, sample_batched in enumerate(self.dataloader):
                sample = sample_batched['features'].view(-1, 1, n_features).type(dtype)
                label_tensor = sample_batched['labels'].type(long_dtype)

                # zero the gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                #print(sample.shape)
                output, _ = self.net(sample)
                loss = criterion(output, label_tensor)
                loss.backward()

                self.optimizer.step()
        

    def score(self, kmers, GT):

        n_features = kmers.shape[1]#4096 #2079
        test_dataset = myDataset(kmers, GT)
        test_dataloader = DataLoader(test_dataset, batch_size=250, shuffle=True, num_workers=2)

        y_true = []
        y_pred = []
        probabilities = []

        with torch.no_grad():
            self.net.eval()
            
            for i_batch, sample_batched in enumerate(test_dataloader):

                sample = sample_batched['features'].view(-1, 1, n_features).type(dtype)
                y_true.extend(sample_batched['labels'].cpu().tolist())

                #calculate the prediction by running through the network
                outputs,_ = self.net(sample)

                #The class with the highest energy is what we choose as prediction
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())

        n = self.n_clusters
        w = np.zeros((n, n), dtype=np.int64)
        for i in range(len(y_true)):
            w[y_true[i], y_pred[i]] += 1
        
        return (np.sum(np.diag(w))/np.sum(w))
    
    def predict(self, kmers, GT=None):
        
        n_features = kmers.shape[1] #4096 #2079
        test_dataset = myDataset(kmers, np.zeros(kmers.shape[0]))
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

        y_pred = []

        with torch.no_grad():
            self.net.eval()
            
            for i_batch, sample_batched in enumerate(test_dataloader):

                sample = sample_batched['features'].view(-1, 1, n_features).type(dtype)
                
                #calculate the prediction by running through the network
                outputs,_ = self.net(sample)

                #The class with the highest energy is what we choose as prediction
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
        return np.array(y_pred)

        
    def rich_predict(self, kmers, GT=None):
        
        n_features = kmers.shape[1] #4096 #2079
        test_dataset = myDataset(kmers, GT)
        test_dataloader = DataLoader(test_dataset, batch_size=250, shuffle=True, num_workers=2)

        y_true = []
        y_pred = []
        probabilities = []

        with torch.no_grad():
            self.net.eval()
            
            for i_batch, sample_batched in enumerate(test_dataloader):

                sample = sample_batched['features'].view(-1, 1, n_features).type(dtype)
                y_true.extend(sample_batched['labels'].cpu().tolist())

                #calculate the prediction by running through the network
                outputs,_ = self.net(sample)

                #The class with the highest energy is what we choose as prediction
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())

        miss = []
        n = self.n_clusters
        w = np.zeros((n, n), dtype=np.int64)

        for i in range(len(y_true)):
            w[y_true[i], y_pred[i]] += 1
            
            if y_true[i] != y_pred[i]:
                miss.append(i)
        
        print(np.sum(np.diag(w))/np.sum(w))

        return w, miss
    

class Contrastive_Model(): 
    def __init__(self, args: dict):
        
        self.sequence_file = args['sequence_file']
        self.GT_file = args['GT_file']

        self.n_clusters = args['n_clusters']
        self.k = args['k']
        
        if args['model_size'] == 'linear':
            self.n_features = 4**self.k
            self.net = SuperNet(self.n_features, args['n_clusters'])
            self.reduce = False
            
        elif args['model_size'] == 'small':
            if self.k % 2 == 0: n_in = (4**self.k + 4**(self.k//2))//2 
            else: n_in = (4**self.k)//2
            #d = {4: 135, 5: 511, 6: 2079}
            self.n_features = n_in
            self.net = SuperNet(self.n_features, args['n_clusters'])
            self.reduce = True
            
        else:
            raise ValueError("Invalid Model Type")
        
        self.net.apply(weights_init)
        self.net.to(device)
        self.epoch = 0
        self.EPS = sys.float_info.epsilon
        
        self.n_mimics = args['n_mimics']
        self.batch_sz = args['batch_sz']
        self.optimizer = args['optimizer']
        self.l = args['lambda'] 
        self.lr = args['lr']
        self.weight = args['weight']
        self.schedule = args['scheduler']
        self.mutate = True

        if self.optimizer == 'RMSprop': 
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=0.01, momentum=0.9)
        elif self.optimizer == 'Adam':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        else:
            raise ValueError("Optimizer not supported")
        
        if self.schedule == 'Plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif self.schedule == 'Triangle':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5, mode="triangular2")
        
    def build_dataloader(self):
        #Data Files
        data_path = self.sequence_file
        GT_file = self.GT_file
                
        self.dataloader = create_dataloader(data_path, 
                                             self.n_mimics, 
                                             k=self.k, 
                                             batch_size=self.batch_sz, 
                                             GT_file=GT_file,
                                             reduce=self.reduce)

    def contrastive_training_epoch(self):
        self.net.train()
        running_loss = 0.0

        for i_batch, sample_batched in enumerate(self.dataloader):
            sample = sample_batched['true'].view(-1, 1, self.n_features).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, self.n_features).type(dtype)
            
            # zero the gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            z1, h1 = self.net(sample)
            z2, h2 = self.net(modified_sample)

            loss = info_nce_loss(h1, h2, 0.90)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss
            
        running_loss /= i_batch

        if self.schedule == 'Plateau':
            self.scheduler.step(running_loss)
        elif self.schedule == 'Triangle':
            self.scheduler.step()
        self.epoch += 1

        return running_loss.item()
    

    def predict(self, data=None):
        
        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file, reduce=self.reduce)
        test_dataloader = DataLoader(test_dataset, 
                             batch_size=self.batch_sz,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

        latent = []

        with torch.no_grad():
            self.net.eval()
            
            for test in test_dataloader:
                
                kmers = test['kmer'].view(-1, 1, self.n_features).type(dtype)
                outputs, logits = self.net(kmers)
                latent.extend(logits.cpu().tolist())
        
        return np.array(latent)


class CLE():
    def __init__(self, sequence_file, n_clusters=200, n_epochs=100, 
                n_mimics=3, batch_sz=512, k=4, weight=0.25, n_voters=1):
        
        self.args = dict()
        self.args['sequence_file'] = sequence_file
        self.args['n_clusters'] = n_clusters

        self.args['n_epochs'] = n_epochs
        self.args['n_mimics'] = n_mimics
        self.args['batch_sz'] = batch_sz

        self.args['GT_file'] = None
        self.args['k'] = k

        self.args['optimizer'] = "RMSprop"
        self.args['lambda'] = 2.8
        self.args['weight'] =  weight
        self.args['n_voters'] = n_voters
        self.args["lr"] = 1e-4
        self.args["model_size"] = "linear"  #5e-4
        self.args['scheduler'] = None
        
    def encode(self):
        model = Contrastive_Model(self.args)
        model.names, model.lengths, model.GT, model.cluster_dis = SummaryFasta(model.sequence_file,
                                                                            model.GT_file)
        
        model.build_dataloader()
        predictions = []


        sys.stdout.write(f"\r........... Training Model ................")
        sys.stdout.flush()
        model.net.apply(weights_init)
        model.epoch = 0
        model_loss = []
        for i in range(self.args['n_epochs']):
            loss = model.contrastive_training_epoch()
            model_loss.append(loss)

        length = len(model.names)
        latent = model.predict()

        return model.names, latent
    

class VAE_Model(nn.Module):

    def __init__(self, n_latent=64, alpha=0.5, k=6,
                 beta=200, dropout=0.2, model_size="linear", 
                 cuda=False):

        # Initialize simple attributes
        self.k = k
        self.usecuda = cuda
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = [256, 256]
        self.n_latent = n_latent
        self.dropout = dropout
        
        if model_size == 'linear':
            self.n_features = 4**self.k
            
            self.reduce = False
            
        elif model_size == 'small':
            if self.k % 2 == 0: n_in = (4**self.k + 4**(self.k//2))//2 
            else: n_in = (4**self.k)//2
        
            self.n_features = n_in
            self.reduce = True

        super(VAE_Model, self).__init__()

        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip([self.n_features] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))
        
        print(self.encoderlayers)
        print(self.encodernorms)
        

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[-1], self.n_latent)
        self.logsigma = nn.Linear(self.nhiddens[-1], self.n_latent)

        # Add first decoding layer
        for nin, nout in zip([self.n_latent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))


        print(self.decoderlayers)
        print(self.decodernorms)

        # Reconstruction (output) layer
        self.outputlayer = nn.Linear(self.nhiddens[0], self.n_features)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()


    def _encode(self, tensor):
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)

        # Note: This softplus constrains logsigma to positive. As reconstruction loss pushes
        # logsigma as low as possible, and KLD pushes it towards 0, the optimizer will
        # always push this to 0, meaning that the logsigma layer will be pushed towards
        # negative infinity. This creates a nasty numerical instability in VAMB. Luckily,
        # the gradient also disappears as it decreases towards negative infinity, avoiding
        # NaN poisoning in most cases. We tried to remove the softplus layer, but this
        # necessitates a new round of hyperparameter optimization, and there is no way in
        # hell I am going to do that at the moment of writing.
        # Also remove needless factor 2 in definition of latent in reparameterize function.
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu, logsigma):
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * torch.exp(logsigma/2)

        return latent

    def _decode(self, tensor):
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        
        kmers_out = reconstruction


        return kmers_out

    def forward(self, kmers):
        mu, logsigma = self._encode(kmers)
        latent = self.reparameterize(mu, logsigma)
        out = self._decode(latent)

        return out, mu, logsigma

    def calc_loss(self, kmers_in, kmers_out, mu, logsigma):

        sse = (kmers_out - kmers_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.n_features
        kld_weight = 1 / (self.n_latent * self.beta)
        #loss = sse * sse_weight + kld * kld_weight
        loss = sse * (0.5/32.0) + kld / (32*200)

        return loss, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0

        if epoch in batchsteps:
            data_loader = DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for sample in data_loader:
            kmers = sample['features'].type(torch.FloatTensor)
            #print(kmers.shape)

            kmers.requires_grad = True

            if self.usecuda:
                kmers = kmers.cuda()

            optimizer.zero_grad()

            _out, mu, logsigma = self(kmers)

            loss, sse, kld = self.calc_loss(kmers, _out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

        

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        
        length = len(new_data_loader.dataset)
        print(length)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = np.empty((length, self.n_latent), dtype=np.float32)

        row = 0
        with torch.no_grad():
            for sample in new_data_loader:
                kmers = sample['features'].type(torch.FloatTensor)
                # Move input to GPU if requested
                if self.usecuda:
                    kmers = kmers.cuda()

                # Evaluate
                out, mu, logsigma = self(kmers)
                #print(mu.shape)

                if self.usecuda:
                    mu = mu.cpu()
                
                #print(mu.shape)

                latent[row: row + mu.shape[0]] = mu
                row += mu.shape[0]
        #print(row)
        assert row == length
        return latent


    def train_model(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=None, logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError(f'Last batch size {last_batchsize} exceeds dataset length {len(dataloader.dataset)}')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs = len(dataloader)
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN latent:', self.n_latent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        return None
    

def make_vae_dataloader(kmers, labels, batchsize=256, destroy=False, cuda=False):

    dataset = myDataset(kmers, labels=labels)     
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, drop_last=False,
                            shuffle=True, num_workers=2, pin_memory=cuda)

    return dataloader

class VAE_encoder():
    def __init__(self, sequence_file, n_epochs=20,
                  batch_size=512, k=4, n_latent=64, 
                  alpha=0.85, beta=100, dropout=0.3,
                  model_size="linear",                   
                  cuda=False):
        
        self.sequence_file = sequence_file
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.n_latent = n_latent
        self.alpha = alpha 
        self.beta = beta
        self.dropout = dropout
        self.model_size = model_size
        self.cuda = cuda
        

    def encode(self):
        
        names, lengths, GT, cluster_dis = SummaryFasta(self.sequence_file)
        #print(len(names))

        model = VAE_Model(n_latent = self.n_latent, alpha=self.alpha, k=self.k,
                 beta=self.beta, dropout=self.dropout, model_size=self.model_size, 
                 cuda=False)
        
        if self.model_size == 'small':
            reduce = True
        else:
            reduce = False

        names, kmers = kmersFasta(self.sequence_file, k=self.k, reduce=reduce)
        #print(kmers.shape)

        dataloader = make_vae_dataloader(kmers, names, batchsize=self.batch_size) #256
        model.train_model(dataloader, nepochs=self.n_epochs)
        latent = model.encode(dataloader)

        return names, latent
        
