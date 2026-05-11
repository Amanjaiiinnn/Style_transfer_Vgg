import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
#from utils.models import VGGEncoder, Decoder
from utils.models import * #import everything from models.py .
from tqdm import tqdm
from torchvision.utils import save_image



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir',       type=str,   default='C:/Users/india/Desktop/NST/content_data',          help='Path to the content dataset')
    parser.add_argument('--style_dir',         type=str,   default='C:/Users/india/Desktop/NST/style_data',            help='Path to the style dataset')
    parser.add_argument('--vgg',               type=str,   default='C:/Users/india/Desktop/NST/vgg_normalised.pth',    help='Pre trained VGG model path')
    parser.add_argument('--experiment',        type=str,   default='experiment1',                                      help='name of the experiment')


    parser.add_argument('--final_size',        type=int,              default=256,                                     help='Size of final image')
    parser.add_argument('--content_size',      type=int,              default=512,                                     help='Size of content image')
    parser.add_argument('--style_size',        type=int,              default=512,                                     help='Size of style image')
    parser.add_argument('--crop',              action='store_true',   default=True,                                    help='Crop image')

    parser.add_argument('--batch_size',        type=int,              default=4,                                       help='Batch size')


    #def arguments for training so that they can be easily modified from command line when running the script.
   
    parser.add_argument('--lr',               type=float,              default=1e-4,                                   help='Learning rate for the optimizer')    
    parser.add_argument('--lr_decay',         type=float,              default=5e-5,                                   help='Learning rate decay for the optimizer')
    parser.add_argument('--epochs',           type=int,                default=1,                                      help='Number of training epochs')


 

    parser.add_argument('--content_weight',   type=float,              default=1.0,                                    help='Content weight')
    parser.add_argument('--style_weight',     type=float,              default=5,                                      help='Style weight')    

    parser.add_argument('--log_interval',     type=int,                default=1,                                      help='Log interval')
    parser.add_argument('--save_interval',    type=int,                default=1,                                      help='Save interval')
    parser.add_argument('--resume',           action='store_true',     default=False,                                  help='Resume training')
    
    parser.add_argument('--decoder_path',     type=str,                default=None,                                   help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path',   type=str,                default=None,                                   help='Path to optimizer checkpoint')
    


    return parser.parse_args()



def main():
    args = parse_arguments()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)


    #save the arguments values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for arg, value in vars(args).items():
            args_file.write(f'{arg}: {value}\n')

    
   
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)   



    content_dataloader = DataLoader(content_dataset,
                                    batch_size=args.batch_size, # to specify how many images to process in one batch during training
                                    shuffle = True,
                                    pin_memory=True, # to speed up data transfer to GPU from CPU
                                    drop_last=True) # to drop the last batch if it's smaller than the specified batch size
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)      



     
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)  
    )

    if args.resume:  #resume training from checkpoint if specified
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path)) #optimizer and decoder states are saved together to ensure that when resuming training, the optimizer continues from the exact state it was in .

    print('Training started...')


    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

#Stores total losses for current epoch.
        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar: #Processes images batch-by-batch.

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)     

            # print( content_batch.shape)
            # print( style_batch.shape)

            c_feats = encoder(content_batch) # VGG extracts features from images. Outputs multiple feature maps from different layers.
            s_feats = encoder(style_batch)

            # to check untill now is model working or not .
            # print(len(c_feats))
            # print(len(s_feats))

            # print(c_feats[0].shape)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1]) #preserve content structure transfer style texture/colors.

            g = decoder(t) # Decoder converts transformed features back into image. TUPLE OF 4 FEATURE MAPS FROM DIFFERENT LAYERS OF VGG, LAST ONE IS THE DEEPEST LAYER (RELU4-1) WHICH CAPTURES HIGH-LEVEL CONTENT STRUCTURE. ADAPTIVE INSTANCE NORMALIZATION MATCHES THE STATISTICS OF THE CONTENT FEATURES TO THE STYLE FEATURES TO TRANSFER STYLE WHILE PRESERVING CONTENT.

            g_feats = encoder(g) #Pass generated image through VGG again.

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight   #Does generated image preserve content structure? 

            loss_s = 0    #Style loss accumulates over layers.
            for g_f, s_f in zip(g_feats, s_feats):   #Compare generated features with style features layer-by-layer. mean and std of style info.
                g_mean, g_std = calc_mean_std(g_f)   # generated features mean and std
                s_mean, s_std = calc_mean_std(s_f)    # style features mean and std
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight #Controls strength of style transfer. Higher: more artistic less original content

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #Optimizer updates decoder parameters.  here epoch part complete

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}') #Shows live training losses.

            running_loss += loss.item()   #accumulate total loss for epoch   . i.e. Used for average epoch metrics.
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step() # Adjust learning rate based on schedule after each epoch. smaller LR later more stable training

        running_loss /= len(content_dataloader) #Average loss per batch.
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0: #Print losses every few epochs.
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}') #Saves trained decoder weights. tqdm make it easy to save intermediate results and monitor training progress without interrupting the flow of the training loop. It allows you to print messages and save checkpoints at specified intervals while still keeping the progress bar intact and updating it in real-time.

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth') #Saves trained decoder weights.
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth') #Saves optimizer state. Useful for resuming training

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0) #content images style images generated images into one tensor.
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size) #Creates image grid.


    
if __name__ == '__main__':
    main()




#  eg train for 150 epooch with default values . then stop 
# then resume and train 100 epochs with style weight 10  size of final image from 256 to 512.