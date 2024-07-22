import torch
 
def euclidean_dist(x, y):
    """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
 
def MaxminNorm(x):
    # 沿着行的方向计算最小值和最大值
    min_vals, _ = torch.min(x, dim=1, keepdim=True)
    max_vals, _ = torch.max(x, dim=1, keepdim=True)
    # print('11', min_vals, max_vals)
    
    # 最小-最大缩放，将x的范围缩放到[0, 1]
    scaled_x = (x - min_vals) / (max_vals - min_vals)
    return scaled_x

if __name__ == '__main__':
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0], [3.0, 1.0, 2.0, 5.0]])
    y = torch.tensor([[3.0, 1.0, 2.0, 5.0], [2.0, 3.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0]])
    # dist_matrix = euclidean_dist(x, y)
    print('1', x.size())
    aa = MaxminNorm(x)
    print(aa.size(), aa)