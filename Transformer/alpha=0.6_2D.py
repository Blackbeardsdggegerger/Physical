import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
import scipy.io
from scipy.special import gamma


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x))) #torch.size(31,32,31)
        out = torch.relu(self.bn2(self.conv2(out))) #torch.size(31,32,31)
        out = torch.relu(self.bn3(self.conv3(out))) #torch.size(31,32,31)
        return out

class ConvEncoder1(nn.Module):
    def __init__(self):
        super(ConvEncoder1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x))) #torch.size(31,32,31)
        out = torch.relu(self.bn2(self.conv2(out))) #torch.size(31,32,31)
        out = torch.relu(self.bn3(self.conv3(out))) #torch.size(31,32,31)
        return out

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x))) #torch.size(31,32,31)
        out = torch.relu(self.bn2(self.conv2(out))) #torch.size(31,32,31)
        out = torch.relu(self.conv3(out)) #torch.size(31,32,31)
        return out

class Transformer(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        # 定义编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=hidden_dim,dim_feedforward=32)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 定义解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=hidden_dim,dim_feedforward=32)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):
        # src: [sequence_length, batch_size, input_dim]
        # tgt: [sequence_length, batch_size, out_dim]
        enc_output = self.encoder(src)  # 编码器输出
        dec_output = self.decoder(tgt, enc_output)  # 解码器输出
        return dec_output

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()

        self.emobing=ConvEncoder()
        self.emobing1 = ConvEncoder1()
        self.Transformer=Transformer(input_dim=32, out_dim=32, hidden_dim=4, num_layers=6)
        self.decoder=ConvDecoder()
        self.net = torch.nn.Sequential(
            nn.Linear(32*20*20, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 400))

    def forward(self,x,tgt):
       src = self.emobing(x)
       lent = src.shape[0]
       lenx = src.shape[2]
       leny = src.shape[3]
       src=src.permute(2,3,0,1)
       src = src.reshape(lenx * leny,lent,-1)

       tgt1=self.emobing1(tgt)
       lent1 = tgt1.shape[0]
       lenx1 = tgt1.shape[2]
       leny1 = tgt1.shape[3]
       tgt1 = tgt1.permute(2,3,0,1)
       tgt1 = tgt1.reshape(lenx1 * leny1, lent1, -1)
       out=self.Transformer(src,tgt1)
       out=out.permute(1,2,0)
       out=out.reshape(lent1,-1,lenx1,leny1)
       out = out.reshape(20, -1)  # flatten the tensor torch.size(31,128)
       out = self.net(out)
       out=out.reshape(-1,1)
       xs = torch.mul(torch.mul(x[:, 1, :, :].reshape(-1, 1).squeeze(), x[:, 2, :, :].reshape(-1, 1).squeeze()),
                      torch.mul(1 - x[:, 1, :, :].reshape(-1, 1).squeeze(), 1 - x[:, 2, :, :].reshape(-1, 1).squeeze()))
       xs = torch.mul(x[:, 0, :, :].reshape(-1, 1).squeeze(), xs)
       out_final = torch.mul(xs, out[:, 0])
       size_out = out_final.shape[0]
       out_final = out_final.reshape(size_out, 1)
       return out_final

def aaa(l, alpha):
    output = (l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output


def fpde(x,tgt, net, M1, M2, N, tau):
    u = net(x,tgt)  # 网络得到的数据

    u_matrix = u.reshape(M1 + 1, N + 1, M2 + 1)
    u_matrix = u_matrix.detach().numpy()
    D_t = np.zeros(((M1 + 1,  N + 1, M2 + 1)))

    for n in range(1, N + 1):
        for i1 in range(1, M1):
            for i2 in range(1, M2):
                D_t[i1, n, i2] = D_t[i1, n, i2] + aaa(0, alpha) * tau ** (-alpha) / gamma(2 - alpha) * u_matrix[i1][n][
                    i2]
                for k in range(1, n):
                    D_t[i1, n, i2] = D_t[i1, n, i2] - (
                                (aaa(n - k - 1, alpha) - aaa(n - k, alpha)) * tau ** (-alpha) / gamma(2 - alpha) *
                                u_matrix[i1][k][i2])
                D_t[i1, n, i2] = D_t[i1, n, i2] - aaa(n - 1, alpha) * tau ** (-alpha) / gamma(2 - alpha) * \
                                 u_matrix[i1][0][i2]
    D_t = D_t.flatten()[:, None]
    D_t = Variable(torch.from_numpy(D_t).float(), requires_grad=False)
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x,tgt)),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    d_x1 = u_tx[:, 1,:,:].reshape(-1, 1)
    d_x2 = u_tx[:, 2,:,:].reshape(-1, 1)

    return D_t - d_x1 - d_x2

def l2_relative_error(output, target):
    # 计算 L2 误差
    error_l2 = torch.norm(output - target, p=2)

    # 计算目标的 L2 范数
    target_l2 = torch.norm(target, p=2)

    # 计算 L2 相对误差
    relative_error = error_l2 / target_l2
    return relative_error

if __name__ == '__main__':
    N = 19
    M1 = 19
    M2 = 19
    alpha = 0.6
    t = np.linspace(0, 1, N + 1)
    x1 = np.linspace(0, 1, M1 + 1)
    x2 = np.linspace(0, 1, M2 + 1)
    tau = t[2] - t[1]
    ms_t, ms_x1, ms_x2 = np.meshgrid(t, x1, x2)
    x1 = np.ravel(ms_x1).reshape(-1, 1)
    x2 = np.ravel(ms_x2).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)

    pt_x1_collocation1 = Variable(torch.from_numpy(x1).float(), requires_grad=True)
    pt_x2_collocation1 = Variable(torch.from_numpy(x2).float(), requires_grad=True)
    pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)
    pt_x1_collocation1 = pt_x1_collocation1.reshape(20, 1, 20, 20)
    pt_x2_collocation1 = pt_x2_collocation1.reshape(20, 1, 20, 20)
    pt_t_collocation1 = pt_t_collocation1.reshape(20, 1, 20, 20)

    Exact1 = (t ** alpha) * (1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2)
    tgt = Variable(torch.from_numpy(Exact1).float(), requires_grad=True).reshape(20, 1, 20, 20)
    net = CNN_Transformer()
    mse_cost_function1 = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    Exact1 = (t ** alpha) * (1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2)
    f1 = gamma(alpha + 1) * (1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2)
    f2 = (t ** alpha) * (1 - x2) * np.sin(x2) * (np.sin(x1) + (x1 - 1) * np.cos(x1))
    f3 = (t ** alpha) * (1 - x1) * np.sin(x1) * (np.sin(x2) + (x2 - 1) * np.cos(x2))
    f = f1 + f2 + f3
    pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
    pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)

    iterations = 4000
    for epoch in range(iterations):

        optimizer.zero_grad()  # 梯度归0
        f_out = fpde(torch.cat([pt_t_collocation1, pt_x1_collocation1, pt_x2_collocation1], 1), tgt, net, M1, M2, N,
                     tau)  # output of f(x,t) 公式（1）
        mse_f_1 = mse_cost_function1(f_out, pt_f_collocation1)

        net_u_in = net(torch.cat([pt_t_collocation1, pt_x1_collocation1, pt_x2_collocation1], 1), tgt)

        errort1 = net_u_in - pt_u_collocation1
        mse_u_1 = mse_cost_function1(net_u_in, pt_u_collocation1)
        error = net_u_in - pt_u_collocation1
        error = error.data.cpu().numpy()
        error_max = (np.abs(error)).max()
        error_L2 = l2_relative_error(net_u_in, pt_u_collocation1)
        error_mean = np.mean(np.abs(error))

        # 将误差(损失)累加起来
        loss = mse_f_1 + error_L2
        MSE = mse_u_1

        # u_error_max = mse_u_1111
        # loss = 0.5*mse_f_1 + 0.5*(mse_u_1+mse_u_2)
        # np.savetxt('loss500.txt', (loss))

        loss.backward()  # 反向传播
        optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        with torch.autograd.no_grad():
            if epoch % 50 == 0:
                print(epoch, "Traning Loss:", loss.data)
                print(epoch, "L2", error_L2)
                print(epoch, "MSE", MSE.data)
                print(epoch, "error max:", error_max)
                print(epoch, "error_mean", error_mean)

    net.eval()

    test_M1 = 19
    test_M2 = 19
    test_N = 19
    t0 = np.linspace(0, 1, test_N + 1)
    x1 = np.linspace(0, 1, test_M1 + 1)
    x2 = np.linspace(0, 1, test_M2 + 1)
    # u_real=t**3*(1-x)*np.sin(x)
    x1_plot, x2_plot = np.meshgrid(x1, x2)
    ms_t, ms_x1, ms_x2 = np.meshgrid(t0, x1, x2)
    x1 = np.ravel(ms_x1).reshape(-1, 1)
    x2 = np.ravel(ms_x2).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)
    pt_x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True)
    pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
    pt_x1 = pt_x1.reshape(20, 1, 20, 20)
    pt_x2 = pt_x2.reshape(20, 1, 20, 20)
    pt_t = pt_t.reshape(20, 1, 20, 20)
    unn_torch = net(torch.cat([pt_t, pt_x1, pt_x2], 1), tgt)
    unn_numpy = unn_torch.data.cpu().numpy()

    u_realfla = np.zeros((x1.shape[0], 1))
    for i in range(x1.shape[0]):
        u_realfla[i, 0] = (t[i, 0] ** alpha) * (1 - x1[i, 0]) * np.sin(x1[i, 0]) * (1 - x2[i, 0]) * np.sin(x2[i, 0])
    u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
    u_real_numpy = u_real_torch.data.cpu().numpy()

    error = np.abs(u_real_numpy - unn_numpy)
    error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(u_real_numpy, ord=2)
    print("error L2:", error_L2)

    unn_matrix = unn_numpy.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)
    u_real_matrix = u_real_numpy.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)
    error_matrix = error.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)

    u_real_t0 = u_real_matrix[:, 10, :]
    unn_t0 = unn_matrix[:, 10, :]
    error_t0 = error_matrix[:, 10, :]

    # 创建一个网格
    x_grid, t_grid = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))

    # Plot predicted values
    plt.figure(figsize=(10, 8))
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(x_grid, t_grid, unn_t0.reshape(20, 20), cmap='RdYlBu_r')
    ax1.set_title('Predicted Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.show()

    # Plot true values
    plt.figure(figsize=(10, 8))
    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(x_grid, t_grid, u_real_t0.reshape(20, 20), cmap='RdYlBu_r')
    ax2.set_title('True Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.show()

    # Plot error values
    plt.figure(figsize=(10, 8))
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(x_grid, t_grid, error_t0.reshape(20, 20), cmap='RdYlBu_r')
    ax3.set_title('Error Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.show()



