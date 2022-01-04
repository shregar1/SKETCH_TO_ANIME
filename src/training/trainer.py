class Trainer():
    def __init__(self,device,dataset_dir,outputs_dir):
        
        #datloaders
        self.train_dataset = ImageDataset(root_dir=dataset_dir+"train",
                                          image_size=256+256)
        self.train_dataloader = DataLoader(self.train_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=0)
        self.val_dataset = ImageDataset(root_dir=dataset_dir+"val",
                                        image_size=512)
        self.val_dataloader = DataLoader(self.val_dataset,batch_size=1,shuffle=True)
        
        #optimizers
        self.G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        #lambda
        self.lambda_L1 = 100
        
        #device
        self.device = device
        
        #output_dir
        self.output_dir = outputs_dir
        
        #loss
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        
    def train_fn(self,D,G):
        print(len(self.train_dataloader))
        for i,(input_image,target_image) in enumerate(tqdm.tqdm(self.train_dataloader)):
            x = input_image.to(self.device)
            y = target_image.to(self.device)
            
            #train_discriminator
            y_fake = G(x)
            D_real = D(x,y)
            D_fake = D(x,y_fake.detach())
            D_real_loss = self.BCE_loss(D_real,torch.ones_like(D_real))
            D_fake_loss = self.BCE_loss(D_fake,torch.zeros_like(D_fake))
            
            #Discriminator loss
            D_loss = (D_real_loss + D_fake_loss)/2
            
            D.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
            
            #train_generator
            y_fake = G(x)
            D_fake = D(x,y_fake)
            G_fake_loss = self.BCE_loss(D_fake,torch.ones_like(D_fake))
            G_L1_loss = self.L1_loss(y_fake,y)
            
            #Total Variational loss
            #reg_loss = 10e-4 * (torch.sum(torch.abs(y_fake[:, :, :, :-1] - y_fake[:, :, :, 1:])) + torch.sum(torch.abs(y_fake[:, :, :-1, :] - y_fake[:, :, 1:, :])))
            
            #Perceptual loss
            P_Loss = PerceptualLoss(self.device)
            perceptual_loss = P_Loss.find(y,y,y_fake)
            
            #color loss
            C_loss = ColorLoss(self.device)
            color_loss = C_loss.find(y,y_fake)
            
            #Generator loss
            G_loss = G_fake_loss + self.lambda_L1*G_L1_loss + perceptual_loss + 10*color_loss
            
            G.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()
            
    def save_examples(self,G,epoch,root_dir):
        print("saving")
        for j,(input_images,target_images) in enumerate(self.val_dataloader):
            x = input_images.to("cuda:0")
            fake_images = G(x)
            G.zero_grad()
            for i in range(len(fake_images)):
                input_image = (np.transpose(input_images[i],(1,2,0))+1)/2
                target_image = (np.transpose(target_images[i],(1,2,0))+1)/2
                fake_image = np.transpose(np.array(fake_images.detach().cpu())[i],(1,2,0))
                fake_image = (fake_image+1)/2
                output_image = np.concatenate([input_image,target_image,fake_image],axis=1)
                file_name = "image_epoch-"+str(epoch)+"_sample-"+str(i)+".png"
                file_path = os.path.join(root_dir,file_name)
                plt.imshow(fake_image)
                plt.imsave(file_path,output_image)
                plt.show()
            break
        return
        
    def fit(self, num_epochs,D,G):
        G.to(self.device)
        D.to(self.device)
        G.train()
        D.train()
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []
        
        for epoch in range(num_epochs):
            print(epoch)
            #train
            self.train_fn(D,G)
            
            #checkpointing
            if epoch % 1 == 0:
                torch.save(G,"Generator_model_perceptual_color.pth")
                torch.save(D,"Discriminator_model_perceptual_color.pth")
            #save examples
            self.save_examples(G,epoch,self.output_dir)
        
        