
def load_disp(self, filename):
    data, scale = pfm_imread(filename)
    data = np.ascontiguousarray(data, dtype=np.float32)
    return data


def disp_sample_by_percent(self, disparity):
    h, w = disparity.shape
    splits = self.sample_percent.split(',')
    valid_indices = np.argwhere(disparity > 0)
    if len(splits) == 2:
        random_percent = random.uniform(float(splits[0]), float(splits[1]))
    else:
        random_percent = float(self.sample_percent)
    sparse_points = int(h * w * random_percent)
    p, vl = valid_indices.shape
    if sparse_points > vl:
        # sample_indices = np.random.choice((sparse_points - 1), size=sparse_points, replace=False)
        sample_pixels = valid_indices.transpose(0, 1)
        sparse_points = vl
    else:
        sample_indices = np.random.choice(vl, size=sparse_points, replace=False)
        sample_pixels = valid_indices[:, sample_indices].transpose(0, 1)
    # print(sample_pixels.shape)
    smask = torch.zeros((h, w), dtype=torch.bool)
    if self.mask is None:
        smask[sample_pixels[:, 0], sample_pixels[:, 1]] = True
        if torch.count_nonzero(smask) != sparse_points:
            exit()
    elif self.mask % 2 == 0:
        mask = self.mask // 2
        for i, j in sample_pixels:
            i_min = max(i - mask, 0)
            i_max = min(i + mask, h)
            j_min = max(j - mask, 0)
            j_max = min(j + mask, w)
            smask[i_min:i_max, j_min:j_max] = True
    elif self.mask % 2 == 1:
        mask = self.mask // 2
        for i, j in sample_pixels:
            i_min = max(i - mask, 0)
            i_max = min(i + mask + 1, h)
            j_min = max(j - mask, 0)
            j_max = min(j + mask + 1, w)
            smask[i_min:i_max, j_min:j_max] = True

    if self.mask == 2 and torch.count_nonzero(smask) > 4 * sparse_points:
        exit()
    elif self.mask == 3 and torch.count_nonzero(smask) > 9 * sparse_points:
        exit()

    # print('smask count', torch.count_nonzero(smask))
    # smask = torch.from_numpy(smask.astype(np.float32))
    sdisparity = disparity * smask
    full_indices = torch.ones((h, w), dtype=torch.bool)
    # indices = torch.nonzero(full_indices)
    indices = torch.nonzero(sdisparity)
    disp_value = sdisparity[indices[:, 0], indices[:, 1]].view(-1, 1)
    sparse_lidar_uvz = torch.cat((indices, disp_value), dim=-1)  # disp_value->z indices->u,v
    return smask, sdisparity, sparse_lidar_uvz