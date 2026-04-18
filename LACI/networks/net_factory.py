from networks.VNet_BCP import VNet_BCP,  VNet


def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train", tsne=0):
    if net_type == "VNet_BCP" and mode == "train" and tsne==0:
        net = VNet_BCP(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet_BCP" and mode == "test" and tsne==0:
        net = VNet_BCP(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if net_type == "VNet" and mode == "train" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num).cuda()
    if net_type == "VNet" and mode == "test" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num).cuda()
    # if net_type == "Vnet_ACMT":
    #     net = VNet_ACMT(n_channels=in_chns, n_classes=class_num,
    #                normalization='batchnorm', has_dropout=True).cuda()
    return net