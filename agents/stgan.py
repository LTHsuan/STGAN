import cv2
import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets import *
from models.stgan import Generator, Discriminator
from utils.misc import print_cuda_statistics
from torchvision import transforms
from PIL import Image
import numpy as np

cudnn.benchmark = True


class Preprocessing:
    def __init__(self, minimum_brightness=1.5):
        self.minimum_brightness = minimum_brightness
        self.brightness = 0

    def caculate_brightness(self, img):
        cols, rows, a = img.shape
        brightness = np.sum(img) / (255 * cols * rows)
        return brightness

    def hisEqulColor(self, img):  # 直方距離放大

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
        cv2.merge(channels, ycrcb)
        img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        return img_eq

    def lightProcessing(self, img , minimum_brightness):
        #self.minimum_brightness = 1.5  # 閥值
        brightness = self.caculate_brightness(img)
        ratio = brightness / minimum_brightness
        # r = cv2.convertScaleAbs(img, alpha = 1 / ratio , beta = 0)
        r = cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)
        return r , brightness

class STGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("STGAN")
        self.logger.info("Creating STGAN architecture...")

        self.G = None
        self.D = None

        if self.config.mode != "generate":
            self.data_loader = globals()['{}_loader'.format(self.config.dataset)](
                self.config.data_root, self.config.mode, self.config.attrs,
                self.config.crop_size, self.config.image_size, self.config.batch_size,
            Data_grouping = { "train_num" : self.config.train_num,"val_num" : self.config.val_num, "test_num" : self.config.test_num,}) # Load data) # Load data

        self.current_iteration = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print(torch.cuda)
            #print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir)

        self.light_preprocessing = Preprocessing()

    def save_checkpoint(self):#save model and weight
        G_state = {
            'state_dict': self.G.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
        }
        D_state  = {
            'state_dict': self.D.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
        }
        G_filename = 'G_{}.pth.tar'.format(self.current_iteration)
        D_filename = 'D_{}.pth.tar'.format(self.current_iteration)
        torch.save(G_state, os.path.join(self.config.checkpoint_dir, G_filename))
        torch.save(D_state, os.path.join(self.config.checkpoint_dir, D_filename))

    def load_checkpoint(self,weight_dir = None,weight_epoch = None):#revise to can input model weight
        if weight_epoch is None and not(self.G is None):
            return
        self.G = Generator(len(self.config.attrs), self.config.g_conv_dim, self.config.g_layers,
                            self.config.shortcut_layers, use_stu=self.config.use_stu,
                            one_more_conv=self.config.one_more_conv)
        self.D = Discriminator(self.config.image_size, len(self.config.attrs), self.config.d_conv_dim,
                            self.config.d_fc_dim, self.config.d_layers)

        if self.config.checkpoint is None and weight_epoch is None:
            self.G.to(self.device)
            self.D.to(self.device)
            return

        if weight_epoch is None:
            G_filename = 'G_{}.pth.tar'.format(self.config.checkpoint)
            D_filename = 'D_{}.pth.tar'.format(self.config.checkpoint)
            G_filename = os.path.join(self.config.checkpoint_dir, G_filename)
            D_filename = os.path.join(self.config.checkpoint_dir, D_filename)
        else:
            G_filename = 'G_{}.pth.tar'.format(weight_epoch)
            D_filename = 'D_{}.pth.tar'.format(weight_epoch)
            G_filename = os.path.join(weight_dir, G_filename)
            D_filename = os.path.join(weight_dir, D_filename)

        G_checkpoint = torch.load(G_filename,map_location=self.device)
        D_checkpoint = torch.load(D_filename,map_location=self.device)

        G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
        D_to_load = {k.replace('module.', ''): v for k, v in D_checkpoint['state_dict'].items()}
        self.current_iteration = self.config.checkpoint
        self.G.load_state_dict(G_to_load)
        self.D.load_state_dict(D_to_load)
        self.G.to(self.device)
        self.D.to(self.device)
        if self.config.mode == 'train':
            self.optimizer_G.load_state_dict(G_checkpoint['optimizer'])
            self.optimizer_D.load_state_dict(D_checkpoint['optimizer'])

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # get hair color indices
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(len(selected_attrs)):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # set one hair color to 1 and the rest to 0
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # reverse attribute value

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def run(self): #Direct run,cahck your train_stgan.yaml
        assert self.config.mode in ['train', 'test', 'generate']
        try:
            if self.config.mode == 'train':
                self.train()
            elif self.config.mode == 'test':
                self.test()
            else:
                self.Generate_imager(image = str(self.config.image_path))
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            print(e)
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
        finally:
            self.finalize()


    def train(self):
        print(f"Train : {self.config.train_num}\nValidation : {self.config.val_num}\nTest : {self.config.test_num}")
        self.load_checkpoint()
        self.optimizer_G = optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.optimizer_D = optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.lr_decay_iters, gamma=0.1)
        self.lr_scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.lr_decay_iters, gamma=0.1)

        if self.cuda and self.config.ngpu > 1:
            self.G = nn.DataParallel(self.G, device_ids=list(range(self.config.ngpu)))
            self.D = nn.DataParallel(self.D, device_ids=list(range(self.config.ngpu)))

        val_iter = iter(self.data_loader.val_loader)
        x_sample, c_org_sample = next(val_iter)
        x_sample = x_sample.to(self.device)
        c_sample_list = self.create_labels(c_org_sample, self.config.attrs)
        c_sample_list.insert(0, c_org_sample)  # reconstruction

        self.g_lr = self.lr_scheduler_G.get_lr()[0]
        self.d_lr = self.lr_scheduler_D.get_lr()[0]

        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        for i in range(self.current_iteration, self.config.max_iters):
            print(f"iters:{self.current_iteration + 1}/{self.config.max_iters}")
            self.G.train()
            self.D.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # fetch real images and labels
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader.train_loader)
                x_real, label_org = next(data_iter)

            # generate target domain labels randomly
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)         # input images
            c_org = c_org.to(self.device)           # original domain labels
            c_trg = c_trg.to(self.device)           # target domain labels
            label_org = label_org.to(self.device)   # labels for computing classification loss
            label_trg = label_trg.to(self.device)   # labels for computing classification loss

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # compute loss with real images
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)#caculate facker loss
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # compute loss with fake images
            attr_diff = c_trg - c_org
            attr_diff = attr_diff * torch.rand_like(attr_diff) * (2 * self.config.thres_int)
            x_fake = self.G(x_real, attr_diff)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # compute loss for gradient penalty
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # backward and optimize
            d_loss_adv = d_loss_real + d_loss_fake + self.config.lambda_gp * d_loss_gp
            d_loss = d_loss_adv + self.config.lambda1 * d_loss_cls
            self.optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            self.optimizer_D.step()

            # summarize
            scalars = {}
            scalars['D/loss'] = d_loss.item()
            scalars['D/loss_adv'] = d_loss_adv.item()
            scalars['D/loss_cls'] = d_loss_cls.item()
            scalars['D/loss_real'] = d_loss_real.item()
            scalars['D/loss_fake'] = d_loss_fake.item()
            scalars['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.config.n_critic == 0:
                # original-to-target domain
                x_fake = self.G(x_real, attr_diff)
                out_src, out_cls = self.D(x_fake)
                g_loss_adv = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # target-to-original domain
                x_reconst = self.G(x_real, c_org - c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # backward and optimize
                g_loss = g_loss_adv + self.config.lambda3 * g_loss_rec + self.config.lambda2 * g_loss_cls
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # summarize
                scalars['G/loss'] = g_loss.item()
                scalars['G/loss_adv'] = g_loss_adv.item()
                scalars['G/loss_cls'] = g_loss_cls.item()
                scalars['G/loss_rec'] = g_loss_rec.item()

            self.current_iteration += 1

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if self.current_iteration % self.config.summary_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print('Elapsed [{}], Iteration [{}/{}]'.format(et, self.current_iteration, self.config.max_iters))
                for tag, value in scalars.items():
                    self.writer.add_scalar(tag, value, self.current_iteration)

            if self.current_iteration % self.config.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    x_sample = x_sample.to(self.device)
                    x_fake_list = [x_sample]
                    for c_trg_sample in c_sample_list:
                        attr_diff = c_trg_sample.to(self.device) - c_org_sample.to(self.device)
                        attr_diff = attr_diff * self.config.thres_int
                        x_fake_list.append(self.G(x_sample, attr_diff.to(self.device)))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    self.writer.add_image('sample', make_grid(self.denorm(x_concat.data.cpu()), nrow=1),
                                          self.current_iteration)
                    save_image(self.denorm(x_concat.data.cpu()),
                               os.path.join(self.config.sample_dir, 'sample_{}.jpg'.format(self.current_iteration)),
                               nrow=1, padding=0)

            if self.current_iteration % self.config.checkpoint_step == 0:
                self.save_checkpoint()

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()

    def test(self):
        # self.load_checkpoint()
        # self.G.to(self.device)
        # self.D.to(self.device)
        #
        # tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
        #                   desc='Testing at checkpoint {}'.format(self.config.checkpoint))
        #
        # self.G.eval()
        # self.D.eval()
        # with torch.no_grad():
        #     for i, (x_real, c_org) in enumerate(tqdm_loader):
        #         x_real = x_real.to(self.device)
        #         print(x_real.size())
        #         c_trg_list = self.create_labels(c_org, self.config.attrs)
        #
        #         x_fake_list = [x_real]
        #         for c_trg in c_trg_list:
        #             attr_diff = c_trg - c_org
        #             x_fake_list.append(self.G(x_real, attr_diff.to(self.device)))#測試單裝影像
        #
        #         x_concat = torch.cat(x_fake_list, dim=3)
        #         result_path = os.path.join(self.config.result_dir, 'sample_{}.jpg'.format(i + 1))
        #         save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

        self.load_checkpoint()
        self.G.to(self.device)
        self.D.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.G.eval()
        self.D.eval()
        d_loss_cls = []
        count = 0
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(tqdm_loader):
                t = transforms.Compose([transforms.Resize(self.config.image_size),
                                        transforms.CenterCrop(self.config.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                        ])
                #
                x_real, brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real), 1.5)
                x_real = self.np_to_PLT(x_real)
                # x_real = self.tensor_to_PIL(x_real)
                #
                x_real = t(x_real).to(self.device)
                x_real = torch.reshape(x_real, (1, x_real.shape[0], x_real.shape[1], x_real.shape[2]))
                out_src, out_cls = self.D(x_real)
                d_loss_real = - torch.mean(out_src)  # caculate facker loss
                d_loss_cls.append( sum(torch.abs(out_cls - c_org))/self.config.batch_size)
                count += 1
        print(count)
        print("d_loss_cls :",sum(d_loss_cls)/count)
    def np_to_PLT(self,image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def PLT_to_np(self,image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def tensor_to_np(self,tensor):
        img = tensor.mul(255).clamp(0, 255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def tensor_to_PIL(self,tensor):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    def Generate_imager(self,image = "",weight_dir="",weight_epoch = None):#Generate one face by auto setting attribute
        '''
        use D to get the org_attrs,and generated face by opposite org_attrs,result save in sample_g.jpg
        input:
            image = path or Image().open(). type:str or PIL.Image.Image
            weight_dir = load model path , default = self.config.checkpoint_dir
            weight_epoch = load model epoch , default = self.config.checkpoint

        output:
            generate imager = use StGAN generate face image(diffent attr). type:PIL.Image.Image
        '''
        self.load_checkpoint(weight_dir,weight_epoch)
        self.G.to(self.device)
        self.D.to(self.device)
        if type(image) == str:
            x_real = Image.open(image).convert('RGB')
        else:
            x_real = image.convert('RGB')



        attrs_org , _ = self.Creat_attrs_org(x_real)  # use D to get attrs_org
        attrs_org= torch.tensor([attrs_org])
        print("attrs_org : ",attrs_org)
        # x_real, brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real), 1.5)
        # x_real = self.np_to_PLT(x_real)
        attrs_trg = attrs_org.clone()  # set attrs_trg = attrs_org

        t = transforms.Compose([transforms.Resize(self.config.image_size),
                            transforms.CenterCrop(self.config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ])

        x_real , brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real),1.5)
        x_real = self.np_to_PLT(x_real)

        x_real = t(x_real).to(self.device)
        x_real = torch.reshape(x_real,(1,x_real.shape[0],x_real.shape[1],x_real.shape[2]))
        self.G.eval()
        self.D.eval()
        x_fake_list = [x_real]

        with torch.no_grad():
            # for i in range(10):
            #     for j in range(1,2):
            #         attrs_trg[0][j] = attrs_org[0][j] + (attrs_org[0][j]*-0.2)*i#change the attributes
            for i in range(len(attrs_org[0])):
                attrs_trg = attrs_org.clone()
                attrs_trg[0][i] = attrs_org[0][i] + (attrs_org[0][i] * -1)  # change the attributes
                attr_diff = attrs_trg - attrs_org #caculate attr_diff
                print("attr_diff : ",attr_diff)
                attr_diff = attr_diff.to(self.device)
                x_fake = self.G(x_real, attr_diff.to(self.device)) # Generate face
                text = self.config.attrs[i]
                x_fake_list.append(x_fake)
                x_concat = torch.cat(x_fake_list, dim=3) # Collage image

                x_concat = self.denorm(x_concat.data.cpu())
                #img = self.tensor_to_np(x_concat)
                # cv2.imshow("resulit",img)
                # cv2.waitKey()

                result_path = f'sample/sample{i}.jpg'
                # x_fake = x_fake.to("cpu")
                # x_fake, _ = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_fake), brightness)
                save_image(self.denorm(x_fake), result_path, nrow=1, padding=0)

        return self.tensor_to_PIL(x_concat)

    def Generate_by_face(self,image,attrs_org,attrs_trg,weight_dir= None,weight_epoch = None):#Generate one face by seeting attribute
        '''
        generated face by setting attrs_org and attrs_trg,result save in sample_g.jpg
        input:
            image = path or Image().open(). type:str or PIL.Image.Image
            attrs_org = original attribute. type:list
            attrs_trg = target attribute. type:list
            weight_dir = load model path , default = self.config.checkpoint_dir
            weight_epoch = load model epoch , default = self.config.checkpoint

        output:
            generate imager = use StGAN generate face image. type:PIL.Image.Image
        '''
        self.load_checkpoint(weight_dir,weight_epoch)
        self.G.to(self.device)

        if type(image) == str:
            x_real = Image.open(image).convert('RGB')
        else:
            x_real = image.convert('RGB')

        attrs_org = torch.torch.FloatTensor(attrs_org)
        attrs_trg = torch.torch.FloatTensor(attrs_trg)

        x_real, brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real), 1.5)
        x_real = self.np_to_PLT(x_real)


        t = transforms.Compose([transforms.Resize(self.config.image_size),
                            transforms.CenterCrop(self.config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ])

        x_real , brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real),1.5)
        x_real = self.np_to_PLT(x_real)
        x_real = t(x_real).to(self.device)
        x_real = torch.reshape(x_real,(1,x_real.shape[0],x_real.shape[1],x_real.shape[2]))
        self.G.eval()

        with torch.no_grad():
            attr_diff = attrs_trg - attrs_org #caculate attr_diff
            print(attr_diff)
            attr_diff = attr_diff.to(self.device)
            x_fake = self.G(x_real, attr_diff.to(self.device)) # Generate face

            x_fake = self.denorm(x_fake.data.cpu())
            # img = self.tensor_to_np(x_fake)
            # cv2.imshow("resulit",img)
            # cv2.waitKey()

            # result_path = 'sample_g.jpg'
            # save_image(x_fake, result_path, nrow=1, padding=0)
            x_fake = self.tensor_to_PIL(x_fake)

        x_fake , brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_fake),1.5)
        x_fake = self.np_to_PLT(x_fake)
        print(brightness)
        return x_fake

    def Creat_attrs_org(self,image,weight_dir= None,weight_epoch = None): #
        '''
               use D to get the org_attrs
               input:
                   image = path or Image().open(). type:str or PIL.Image.Image
                   weight_dir = load model path , default = self.config.checkpoint_dir
                   weight_epoch = load model epoch , default = self.config.checkpoint
               output:
                   attrs_org = use D generate face image attribute.  type:list
        '''

        self.load_checkpoint(weight_dir,weight_epoch)
        self.D.to(self.device)

        if type(image) == str:
            x_real = Image.open(image).convert('RGB')
        else:
            x_real = image.convert('RGB')

        t = transforms.Compose([transforms.Resize(self.config.image_size),
                            transforms.CenterCrop(self.config.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            ])

        x_real, brightness = self.light_preprocessing.lightProcessing(self.PLT_to_np(x_real), 1.5)
        x_real = self.np_to_PLT(x_real)

        x_real = t(x_real).to(self.device)
        x_real = torch.reshape(x_real,(1,x_real.shape[0],x_real.shape[1],x_real.shape[2]))

        out_src, attrs_org = self.D(x_real)  # use D to get attrs_org
        attrs_org = torch.nn.functional.normalize(attrs_org,2) # Normalizate to -1 and 1
        attrs_org = attrs_org.tolist()[0]# tranform the tensor to list


        # origin_image = self.np_to_PLT(origin_image)
        origin_image = self.denorm(x_real[0].data.cpu())
        origin_image = self.tensor_to_PIL(origin_image)

        # origin_image = self.light_preprocessing.lightProcessing(self.PLT_to_np(origin_image))
        # origin_image = self.np_to_PLT(origin_image)

        origin_image , _ = self.light_preprocessing.lightProcessing(self.PLT_to_np(origin_image),brightness)
        origin_image = self.np_to_PLT(origin_image)
        return attrs_org , origin_image


    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()
