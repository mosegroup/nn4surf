# <<< import external modules <<<
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# === import external modules ===

# <<< import nn4surf modules <<<
from src.dataloader import TabulatedSeries
from src.utils import save_args
from src.classes import convmodel
from src.argparser import Parser
# === import nn4surf modules ===

    
def main():
    # Main training function
    
    arg_parser = Parser()
    args = arg_parser.parse_args()

    master_folder = args.output_folder

    if os.path.exists(master_folder):
        if os.listdir(master_folder) != []:
            raise OSError(f'Folder "{master_folder}" is not empty. Please, remove it, empty it or change output folder to proceed.')

    try:
        os.mkdir(f'{master_folder}')
    except:
        pass
    
    try:
        os.mkdir(f'{master_folder}/pt_files')
    except:
        pass

    os.system( f'cp train.py {master_folder}/' )
    save_args(f'{master_folder}/args.txt', args)
    
    model = convmodel(
        kernel_size     = args.kernel_size,
        depth           = args.depth,
        channels        = args.channels,
        activation      = torch.nn.Tanh()
        )
    
    model.to(args.device)
    
    if args.model_path != 'None':
        model.load_state_dict(torch.load(args.model_path))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    seed = 1
    epochs = 3000

    train_losses = []
    valid_losses = []

    print('The number of parameters in the model is:', sum(p.numel() for p in model.parameters() if p.requires_grad) )

    train_table_path = args.train_set
    valid_table_path = args.valid_set
        
    train_set = TabulatedSeries( train_table_path, every=20 )
    valid_set = TabulatedSeries( valid_table_path, every=20 )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size      = 512,
        shuffle         = True, # <- in the case of testing and validation, we want a fixed order for data
        num_workers     = args.nproc
        )
    
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set,
        batch_size      = 128,
        shuffle         = False, # <- in the case of testing and validation, we want a fixed order for data
        num_workers     = args.nproc
        )

    for epoch in range(epochs):
        
        model.zero_grad()

        # <<< train loop <<<

        train_losses_epoch = []

        for j, (profile, elastic_mu, _) in enumerate(train_dataloader):

            if j >= 5 and args.debug:
                print('DEBUG MODE ---> breaking')
                break
            
            profile = profile.to(args.device)
            elastic_mu = elastic_mu.to(args.device)
            
            optimizer.zero_grad() #zero-outthe gradients
            
            mu_pred = model(profile)
            
            loss = loss_fn(mu_pred, elastic_mu)

            train_losses_epoch.append( loss.item() )

            loss.backward() #To backpropagate the error all we have to do is to call the backward mmthod
        
            optimizer.step() #Does the update

            if j % 10 == 0:
                print(f'Passing example [{j}]/[{len(train_dataloader)}] in epoch {epoch} \t loss:{loss.item()}')

        train_losses.append( np.mean(train_losses_epoch) )

        # === train loop ===


        # <<< validation loop <<<
        valid_losses_epoch = []

        with torch.no_grad():

            for profile, elastic_mu, x in valid_dataloader:

                elastic_mu = elastic_mu.to(args.device)
                profile = profile.to(args.device)

                mu_pred = model(profile)
                loss = loss_fn(mu_pred, elastic_mu)

                valid_losses_epoch.append( loss.item() )

                if args.debug: break

            valid_losses.append( np.mean(valid_losses_epoch) )

        print(f'Epoch [{epoch}]/[{epochs}], training loss: {train_losses[-1]}, validation loss: {valid_losses[-1]}')
        
        with open(f'{master_folder}/loss.txt', 'a+') as loss_file:
            loss_file.write(f'{train_losses[-1]} {valid_losses[-1]}\n')
        # === validation loop ===

        # save model
        torch.save(model.state_dict(), f'{master_folder}/pt_files/model_{epoch}.pt')
        
        # <<< graphical stuff <<<
        if args.graphics:
            for num_sample in range(10):
                # this prints some examples to have a visual assessment of the training (NOT randomized)
                try:
                    os.mkdir(f'{master_folder}/epoch_{epoch}')
                except:
                    pass

                plt.plot(x[num_sample,0,...], mu_pred[num_sample,0,...].detach().cpu(), label='Predicted energy')
                plt.plot(x[num_sample,0,...], elastic_mu[num_sample,0,...].detach().cpu(), label='True energy')
                plt.legend()
                plt.savefig(f'{master_folder}/epoch_{epoch}/rho_{num_sample}.png')
                plt.close()

                plt.plot(x[num_sample,0,...], profile[num_sample,0,...].detach().cpu(), label='Profile')
                plt.legend()
                plt.savefig(f'{master_folder}/epoch_{epoch}/profile_{num_sample}.png')
                plt.close()
        # === graphical stuff ===


if __name__ == '__main__':
    '''
    Run main function
    '''
    main()
