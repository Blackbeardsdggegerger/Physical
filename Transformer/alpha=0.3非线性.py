import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

from scipy.special import gamma


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))  # torch.size(31,32,31)
        out = torch.relu(self.bn2(self.conv2(out)))  # torch.size(31,32,31)
        out = torch.relu(self.bn3(self.conv3(out)))  # torch.size(31,32,31)
        return out


class ConvEncoder1(nn.Module):
    def __init__(self):
        super(ConvEncoder1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))  # torch.size(31,32,31)
        out = torch.relu(self.bn2(self.conv2(out)))  # torch.size(31,32,31)
        out = torch.relu(self.bn3(self.conv3(out)))  # torch.size(31,32,31)
        return out


class Transformer(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        # 定义编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=hidden_dim, dim_feedforward=32)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 定义解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=hidden_dim, dim_feedforward=32)
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

        self.emobing = ConvEncoder()
        self.emobing1 = ConvEncoder1()
        self.Transformer = Transformer(input_dim=32, out_dim=32, hidden_dim=4, num_layers=6)
        # self.decoder=ConvDecoder()
        self.net = torch.nn.Sequential(
            nn.Linear(32 * 31, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 31))

    def forward(self, x, tgt):
        src = self.emobing(x)  # torch.size(31,32,31)
        src = src.permute(2, 0, 1)  # torch.size(31,31,32)
        tgt = self.emobing1(tgt)  # torch.size(31,32,31)
        tgt = tgt.permute(2, 0, 1)  # torch.size(31,31,32)
        out = self.Transformer(src, tgt)  # torch.size(31,31,32)
        out = out.permute(1, 2, 0)  # torch.size(31,32,31)
        # out=self.decoder(out)
        out = out.reshape(31, -1)  # flatten the tensor torch.size(31,128)
        out = self.net(out)
        out = out.reshape(-1, 1)
        xs = torch.mul(x[:, 0, :].reshape(-1, 1).squeeze() ** alpha,
                       torch.mul(x[:, 1, :].reshape(-1, 1).squeeze(), 1 - x[:, 1, :].reshape(-1, 1).squeeze()))
        out_final = torch.mul(xs, out[:, 0])
        size_out = out_final.shape[0]
        out_final = out_final.reshape(size_out, 1)
        return out_final


def aaa(l, alpha):
    output = (l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output


def fpde(x, tgt, net, M, N, tau):
    u = net(x, tgt)  # 网络得到的数据
    u_matrix = u.reshape(M + 1, N + 1)
    u_matrix = u_matrix.detach().numpy()
    # u_matrix = u_matrix**alpha
    D_t = np.zeros(((M + 1, N + 1)))

    for n in range(1, N + 1):
        for i in range(1, M):
            D_t[i, n] = D_t[i, n] + aaa(0, alpha) * tau ** (-alpha) / gamma(2 - alpha) * u_matrix[i][n]
            for k in range(1, n):
                D_t[i, n] = D_t[i, n] - (
                            (aaa(n - k - 1, alpha) - aaa(n - k, alpha)) * tau ** (-alpha) / gamma(2 - alpha) *
                            u_matrix[i][k])
            D_t[i, n] = D_t[i, n] - aaa(n - 1, alpha) * tau ** (-alpha) / gamma(2 - alpha) * u_matrix[i][0]
    D_t = D_t.flatten()[:, None]
    D_t = Variable(torch.from_numpy(D_t).float(), requires_grad=False)
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x, tgt)),
                               create_graph=True, allow_unused=True)[0]
    d_x = u_tx[:, 1, :]
    d_x = d_x.reshape(-1, 1)
    uuu = torch.mul(torch.mul(u, (1 - u)), (u - 1))
    size_uuu = uuu.shape[0]
    uuu = uuu.reshape(size_uuu, 1)

    return D_t - d_x - uuu


def l2_relative_error(output, target):
    # 计算 L2 误差
    error_l2 = torch.norm(output - target, p=2)

    # 计算目标的 L2 范数
    target_l2 = torch.norm(target, p=2)

    # 计算 L2 相对误差
    relative_error = error_l2 / target_l2
    return relative_error


if __name__ == '__main__':
    alpha = 0.3
    iterations = 4000
    M = 30
    N = 30
    t = np.linspace(0.0001, 1, N + 1)
    x = np.linspace(0, 1, M + 1)
    tau = t[2] - t[1]
    ms_t, ms_x = np.meshgrid(t, x)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)
    pt_x_collocation1 = Variable(torch.from_numpy(x).float(), requires_grad=True)  # torch.Size([31, 1, 31])
    pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)  # torch.Size([31, 1, 31])
    pt_x_collocation1 = pt_x_collocation1.reshape(31, 1, 31)
    pt_t_collocation1 = pt_t_collocation1.reshape(31, 1, 31)
    model = CNN_Transformer()
    mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
    mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    f = np.zeros((x.shape[0], 1))
    Exact1 = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        Exact1[i, 0] = t[i, 0] ** alpha * np.sin(np.pi * x[i, 0])
        f[i, 0] = gamma(alpha + 1) * np.sin(np.pi * x[i, 0]) - np.pi * t[i, 0] ** alpha * np.cos(np.pi * x[i, 0]) - t[
            i, 0] ** alpha * np.sin(np.pi * x[i, 0]) * (t[i, 0] ** alpha * np.sin(np.pi * x[i, 0]) - 1) * (
                          1 - t[i, 0] ** alpha * np.sin(np.pi * x[i, 0]))

    pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
    pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)
    tgt = pt_u_collocation1.reshape(31, 1, 31)
    for epoch in range(iterations):
        optimizer.zero_grad()  # 梯度归0

        # 将变量x,t带入公式（1）
        f_out = fpde(torch.cat([pt_t_collocation1, pt_x_collocation1], 1), tgt, model, M, N,
                     tau)  # output of f(x,t) 公式（1）

        mse_f_1 = mse_cost_function1(f_out, pt_f_collocation1)
        net_u_in = model(torch.cat([pt_t_collocation1, pt_x_collocation1], 1), tgt)
        mse_u_1 = l2_relative_error(net_u_in, pt_u_collocation1)
        mse_u_2 = mse_cost_function1(net_u_in, pt_u_collocation1)
        error = net_u_in - pt_u_collocation1
        error = error.data.cpu().numpy()
        error_max = (np.abs(error)).max()
        error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(Exact1, ord=2)
        error_mean = np.mean(np.abs(error))

        # 将误差(损失)累加起来
        # loss1 = mse_f_1
        MSE = mse_u_2
        # u_error_max = mse_u_1111
        loss = mse_u_1 + mse_f_1

        loss.backward()  # 反向传播
        optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        with torch.autograd.no_grad():
            if epoch % 50 == 0:
                print(epoch, "Traning Loss:", loss.data)
                print(epoch, "L2", error_L2)
                print(epoch, "MSE", MSE.data)
                print(epoch, "error max:", error_max)
                print(epoch, "error_mean", error_mean)

    # 获取预测值、真实值和误差值
    model.eval()  # 切换到评估模式

    # 生成新的测试数据
    test_M = 31
    test_N = 31
    x0 = np.linspace(0, 1, test_M)
    t0 = np.linspace(0.0000001, 1, test_N)
    # u_real=t**3*(1-x)*np.sin(x)

    ms_t, ms_x = np.meshgrid(t0, x0)
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
    pt_x = pt_x.reshape(31, 1, 31)
    pt_t = pt_t.reshape(31, 1, 31)
    output = model(torch.cat([pt_t, pt_x], 1), tgt)
    output_np = output.detach().numpy()  # 获取预测值并转换为numpy
    u_np = pt_u_collocation1.detach().numpy()  # 获取真实值并转换为numpy
    error_np = np.abs(output_np - u_np)  # 计算误差

    # 创建一个网格
    x_grid, t_grid = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, 1, 31), tgt)

    # 绘制3D图像
    fig = plt.figure(figsize=(18, 6))

    # 预测值
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(x_grid, t_grid, output_np.reshape(31, 31), cmap='RdYlBu_r')
    ax1.set_title('Predicted Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')

    # 真实值
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(x_grid, t_grid, u_np.reshape(31, 31), cmap='RdYlBu_r')
    ax2.set_title('True Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')

    # 误差值
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(x_grid, t_grid, error_np.reshape(31, 31), cmap='RdYlBu_r')
    ax3.set_title('Error Values', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_xlabel('x', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_ylabel('t', style='italic', fontsize=20, fontweight='bold', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.show()






