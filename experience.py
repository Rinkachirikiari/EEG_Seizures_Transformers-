import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import random
import time
import datetime
import mne

from torch.utils.data import Dataset

import torch

from torch.autograd import Variable

from torch import nn
# from common_spatial_pattern import csp

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from model import convolution, transformers


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, nb_channels =23, depth=6, n_classes=2, **kwargs):
        super().__init__(

            convolution.PatchEmbedding(emb_size, nb_channels),
            transformers.TransformerEncoder(depth, emb_size),
            transformers.ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50) # sert à quoi ? 
        self.nSub = nsub

        self.start_epoch = 0
        self.root = "C:/Users/MAREZ10/OneDrive - Université Laval/Bureau/Projet Transformers/eeg_data" #Chemin de la donnée sur mon PC

        self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
    
        self.criterion_cls = torch.nn.BCELoss().cuda() # Fonction loss binaire

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def separate_data_intervals(file_str, seizure_presence):
        # Durée de chaque intervalle en secondes (5 minutes)
        interval_duration = 5 * 60

        raw = mne.io.read_raw_edf(file_str)
        #data, times = raw[:, :]
        total_duration = raw.times[-1] # en secondes

        # Nombre total d'intervalle de 10 minutes
        num_intervals = int(total_duration / interval_duration)
        labels = []
        data = []

        # Diviser les données en intervalles de 10 minutes
        for i in range(num_intervals):
            # Calculer le temps de début et de fin de chaque intervalle
            start_time = i * interval_duration
            end_time = (i + 1) * interval_duration

            # Convertir le temps en indice
            start_idx = raw.time_as_index(start_time)
            end_idx = raw.time_as_index(end_time)

            # Extraire les données de l'intervalle
            interval_data, interval_times = raw[:, start_idx:end_idx]
            data.append(interval_data)
            if file_str in seizure_presence.keys():
                is_seizure = False
                for start_seizure, end_seizure in seizure_presence[file_str]:
                    if (start_seizure >= start_time and start_seizure <= end_time) or (end_seizure >= start_time and end_seizure <= end_time) or (start_seizure <= start_time and end_seizure >= end_time):
                        is_seizure = True
                        break
                if is_seizure:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(0)
        return data, labels
        

    def train(self, train_data, train_label, test_data, test_label):

        
        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label - 1)

        dataset = torch.utils.data.TensorDataset(train_data, train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (train_data, train_label) in enumerate(self.dataloader):

                train_data = Variable(train_data.cuda().type(self.Tensor))
                train_label = Variable(train_label.cuda().type(self.LongTensor))

                tok, outputs = self.model(train_data)

                loss = self.criterion_cls(outputs, train_label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == train_label).cpu().numpy().astype(int).sum()) / float(train_label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred


        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred

def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
