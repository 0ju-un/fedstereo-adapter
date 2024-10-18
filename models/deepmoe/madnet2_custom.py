import torch
import torch.nn.functional as F

from models.madnet2 import MADNet2
from models.madnet2.corr import CorrBlock1D
from .moe import feature_extraction_moe
from .utils import ShallowEmbeddingNetwork

class CustomMadNet2(MADNet2):
    def __init__(self, args):
        super().__init__(args)

        a=1
        self.feature_extraction = feature_extraction_moe()

        dim = 128
        self.embedding = ShallowEmbeddingNetwork(dim, 3, cifar=False)
        a=1

    def forward(self, image2, image3, mad=False):
        """ Estimate optical flow between pair of frames """

        embed2 = self.embedding(image2) #!DEBUG
        embed3 = self.embedding(image3)

        im2_fea = self.feature_extraction(image2, mad, embed2)
        im3_fea = self.feature_extraction(image3, mad, embed3)

        if len(im2_fea) == 8: #!DEBUG
            im2_gat = im2_fea[-1]
            im3_gat = im3_fea[-1]
            im2_fea = im2_fea[:-1]
            im3_fea = im3_fea[:-1]
        assert len(im2_fea) == len(im3_fea) == 7

        corr_block = CorrBlock1D

        corr_fn6 = corr_block(im2_fea[6], im3_fea[6], radius=2, num_levels=1)
        corr_fn5 = corr_block(im2_fea[5], im3_fea[5], radius=2, num_levels=1)
        corr_fn4 = corr_block(im2_fea[4], im3_fea[4], radius=2, num_levels=1)
        corr_fn3 = corr_block(im2_fea[3], im3_fea[3], radius=2, num_levels=1)
        corr_fn2 = corr_block(im2_fea[2], im3_fea[2], radius=2, num_levels=1)

        coords0, coords1_6 = self.initialize_flow(im2_fea[6])
        coords0, coords1_5 = self.initialize_flow(im2_fea[5])
        coords0, coords1_4 = self.initialize_flow(im2_fea[4])
        coords0, coords1_3 = self.initialize_flow(im2_fea[3])
        coords0, coords1_2 = self.initialize_flow(im2_fea[2])

        corr6 = corr_fn6(coords1_6)
        disp6 = self.decoder6(torch.cat((im2_fea[6],corr6), 1))
        disp6_u = F.interpolate(disp6 if not mad else disp6.detach(), scale_factor=2)*20./32

        coords1_5 = coords1_5 + disp6_u
        corr5 = corr_fn5(coords1_5)
        disp5 = self.decoder5(torch.cat((im2_fea[5],corr5,disp6_u), 1))
        disp5_u = F.interpolate(disp5 if not mad else disp5.detach(), scale_factor=2)*20./16

        coords1_4 = coords1_4 + disp5_u
        corr4 = corr_fn4(coords1_4)
        disp4 = self.decoder4(torch.cat((im2_fea[4],corr4,disp5_u), 1))
        disp4_u = F.interpolate(disp4 if not mad else disp4.detach(), scale_factor=2)*20./8

        coords1_3 = coords1_3 + disp4_u
        corr3 = corr_fn3(coords1_3)
        disp3 = self.decoder3(torch.cat((im2_fea[3],corr3,disp4_u), 1))
        disp3_u = F.interpolate(disp3 if not mad else disp3.detach(), scale_factor=2)*20./4

        coords1_2 = coords1_2 + disp3_u
        corr2 = corr_fn2(coords1_2)
        disp2 = self.decoder2(torch.cat((im2_fea[2],corr2,disp3_u), 1))

        return disp2, disp3, disp4, disp5, disp6, [im2_gat, im3_gat]


    def compute_loss(self, image2, image3, predictions, gt, validgt, adapt_mode='full', idx=-1, gates=None, lambda_val=0.001):
        assert gates is not None

        ## self_supervised_loss
        # if adapt_mode == 'full':
        #     loss =  [self_supervised_loss(predictions[0], image2, image3),
        #             self_supervised_loss(predictions[1], image2, image3),
        #             self_supervised_loss(predictions[2], image2, image3),
        #             self_supervised_loss(predictions[3], image2, image3),
        #             self_supervised_loss(predictions[4], image2, image3)]
        #     self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()
        #     loss = sum(loss).mean()

            # legacy from original MADNet training (classical average reduction without any weights gives almost identical results)
        loss =  [0.001*F.l1_loss(predictions[0][validgt>0], gt[validgt>0], reduction='sum') / 20.,
                 0.001*F.l1_loss(predictions[1][validgt>0], gt[validgt>0], reduction='sum') / 20.,
                 0.001*F.l1_loss(predictions[2][validgt>0], gt[validgt>0], reduction='sum') / 20.,
                 0.001*F.l1_loss(predictions[3][validgt>0], gt[validgt>0], reduction='sum') / 20.,
                 0.001*F.l1_loss(predictions[4][validgt>0], gt[validgt>0], reduction='sum') / 20.]
        self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()

        gate_loss = sum(torch.linalg.norm(g, 1) for g in gates)

        loss = sum(loss).mean() + lambda_val * gate_loss

        ## code for modular update
        # elif adapt_mode == 'mad':
        #     loss = self_supervised_loss(predictions[idx], image2, image3)
        #
        # elif adapt_mode == 'mad++':
        #     loss = F.l1_loss(predictions[idx][validgt>0], gt[validgt>0])
        #
        # if 'mad' in adapt_mode:
        #     self.update_sample_distribution(idx,loss.cpu(),adapt_mode)

        return loss


    # def compute_loss(self, image2, image3, predictions, gt, validgt, adapt_mode='full', idx=-1):

    #     # self_supervised_loss
    #     if adapt_mode == 'full':
    #         loss =  [self_supervised_loss(predictions[0], image2, image3),
    #                  self_supervised_loss(predictions[1], image2, image3),
    #                  self_supervised_loss(predictions[2], image2, image3),
    #                  self_supervised_loss(predictions[3], image2, image3),
    #                  self_supervised_loss(predictions[4], image2, image3)]
    #         self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()
    #         loss = sum(loss).mean()

    #     elif adapt_mode == 'full++':
    #         # legacy from original MADNet training (classical average reduction without any weights gives almost identical results)
    #         loss =  [0.001*F.l1_loss(predictions[0][validgt>0], gt[validgt>0], reduction='sum') / 20.,
    #                  0.001*F.l1_loss(predictions[1][validgt>0], gt[validgt>0], reduction='sum') / 20.,
    #                  0.001*F.l1_loss(predictions[2][validgt>0], gt[validgt>0], reduction='sum') / 20.,
    #                  0.001*F.l1_loss(predictions[3][validgt>0], gt[validgt>0], reduction='sum') / 20.,
    #                  0.001*F.l1_loss(predictions[4][validgt>0], gt[validgt>0], reduction='sum') / 20.]
    #         self.accumulated_loss += torch.stack([loss[i] * self.loss_weights[i] for i in range(len(loss))],0).detach().cpu()
    #         loss = sum(loss).mean()

    #     # code for modular update
    #     elif adapt_mode == 'mad':
    #         loss = self_supervised_loss(predictions[idx], image2, image3)
        
    #     elif adapt_mode == 'mad++':
    #         loss = F.l1_loss(predictions[idx][validgt>0], gt[validgt>0])
        
    #     if 'mad' in adapt_mode:
    #         self.update_sample_distribution(idx,loss.cpu(),adapt_mode)

    #     return loss

