from __future__ import print_function
from __future__ import division

import sys
sys.path.append('../src/train')
from vvtk_net.v1.layers_pytorch import *

class GPermute(nn.Module):
    def __init__(self, ch_in, groups):
        super(GPermute, self).__init__()
        self.groups = groups
        I = np.eye(groups)
        P = I[np.random.permutation(groups)]
        self.register_buffer('P', torch.from_numpy(np.argmax(P, 1)).long())

    def forward(self, x):                               # (b, c, t)
        b, c, t = x.size()
        x = x.view(b, self.groups, -1, t)               # (b, g, cg, t)
        x = x[:,self.P]     
        x = x.view(b, c, t).contiguous()
        return x

class SpatialGroupEnhance1d(nn.Module):
    def __init__(self, groups=64):
        super(SpatialGroupEnhance1d, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1))
        self.bias     = nn.Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x):                               # (b, c, t)
        b, c, t = x.size()
        x = x.view(b * self.groups, -1, t)              # (b*g, cg, t)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)                # (b*g, 1, t)

        a = xn.view(b * self.groups, -1)                # (b*g, t)
        a = a - a.mean(dim=1, keepdim=True)
        std = a.std(dim=1, keepdim=True) + 1e-5
        a = a / std
        a = a.view(b, self.groups, t)                   # (b, g, t)
        a = a * self.weight + self.bias
        a = a.view(b * self.groups, 1, t)               # (b*g, 1, t)
        x = x * self.sig(a)                             # (b*g, cg, t)
        x = x.view(b, c, t)                             # (b, c, t)
        return x

class DerivConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=9, padding=None, dilation=1, stride=1, bias=False, nb_deriv=8):
        super(DerivConv1d, self).__init__()
        self.xblock1 = CConv1d(ch_in, ch_in*(nb_deriv-1), kernel_size=kernel_size, padding=(kernel_size//2),stride=1, groups=ch_in)
        self.xblock2 = CConv1d(nb_deriv*ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=nb_deriv)

    def forward(self, x):
        x0 = x
        x = self.xblock1(x)
        x = torch.cat([x0, x], 1)
        x = self.xblock2(x)
        return x


class Permute(nn.Module):
    def __init__(self,ch_in):
        super(Permute, self).__init__()
        I = np.eye(ch_in)
        P = I[np.random.permutation(ch_in)]
        self.register_buffer('P', torch.from_numpy(np.argmax(P, 1)).long())

    def forward(self, x):
        x = x[:,self.P]
        return x

class Conv1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=None, dilation=1, stride=1, bias=False, groups=64, permute=True):
        super(Conv1d, self).__init__()
        self.permute = permute
        if self.permute:
            self.p1  = Permute(ch_in)
        self.conv1 = CConv1d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=bias, groups=groups)

    def forward(self, x):
        if self.permute:
            x = self.p1(x)
        x = self.conv1(x)
        return x

class GConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=None, dilation=1, stride=1, bias=False, groups=64, permute=True):
        super(GConv1d, self).__init__()
        self.permute = permute
        if self.permute:
            self.p1  = GPermute(ch_in, groups)
        self.conv1 = CConv1d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=bias, groups=groups)

    def forward(self, x):
        if self.permute:
            x = self.p1(x)
        x = self.conv1(x)
        return x

class col1d(nn.Module):
    def __init__(self, ch_in, ch_out, p_drop=0., stride=1, dilation=1, kernel_size=3):
        super(col1d, self).__init__()
        self.p_drop = p_drop
        # -----------------------
        self.bn1 = nn.BatchNorm1d(ch_in)
        self.pr1 = nn.PReLU(ch_in)
        self.conv1 = Conv1d(ch_in, ch_out, permute=False, kernel_size=kernel_size, padding=(kernel_size//2)*dilation, stride=stride, dilation=dilation, bias=False) #original groups by default 
        # -----------------------
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.pr2 = nn.PReLU(ch_out)
        self.conv2 = Conv1d(ch_out, ch_out, permute=False, kernel_size=1, padding=0, stride=1, bias=False, groups=8) #original groups=8
        self.sge = SpatialGroupEnhance1d(groups=8) #original groups=8

    def forward(self, x):
        x = self.bn1(x)
        x = self.pr1(x)
        x = self.conv1(x)
        # -----------------------
        x = self.bn2(x)
        x = self.pr2(x)
        x = self.conv2(x)
        x = self.sge(x)
        # -----------------------
        if self.p_drop > 0:
            x = F.dropout(x, p=self.p_drop, training=self.training)
        return x


class wrn_layer(nn.Module):
    def __init__(self, ch_in, ch_out, p_drop=0., stride=1, dilation=1, kernel_size=3):
        super(wrn_layer, self).__init__()
        self.p_drop = p_drop
        self.stride = stride
        self.conv_shortcut = ch_in != ch_out  or stride > 1 # conv shortcut (only to adjust nb_ch)
        # -----------------------
        self.col1 = col1d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, p_drop=p_drop) #original no p_drop
        # -----------------------
        self.conv3 = Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False, groups=4) #original groups=4
        self.conv41 = Conv1d(2*ch_out, ch_out, kernel_size=1, padding=0, stride=1, bias=False, groups=8, permute=False) #original groups=8
        self.conv42 = Conv1d(2*ch_out, ch_out, kernel_size=1, padding=0, stride=1, bias=False, groups=8, permute=False) #original groups=8
        self.bn41 = nn.BatchNorm1d(ch_out)
        self.bn42 = nn.BatchNorm1d(ch_out)
        self.p1  = Permute(ch_out)

    def forward(self, x):
        x0 = x
        # -----------------------
        x = self.col1(x)             
        # ----------------------- 
        x0 = self.conv3(x0)
        c1 = self.conv41( torch.cat( [x, x0], 1) )
        c1 = self.bn41(c1)
        c1 = sigmoid(c1)
        c2 = self.conv42( torch.cat( [x, x0], 1) )
        c2 = self.bn42(c2)
        c2 = sigmoid(c2)
        x = c1 * x0 + c2 * x
        x = self.p1(x)
        if self.stride > 1:
            x = apool(x, self.stride)
        return x


def wrn_block(ch_in, ch_out, nb_blocks, p_drop=0., stride=1, dilation=1, kernel_size=3):
    layers = []
    for _ in range(nb_blocks):
        layers.append(wrn_layer(ch_in, ch_out, p_drop, stride, kernel_size=kernel_size,dilation=dilation))
        ch_in = ch_out
        stride = 1
        dilation=1
        kernel_size=3
    return nn.Sequential(*layers)

class WideResNet(nn.Module):
    def __init__(self, input_dim, output_dim, n=3, widen_factor=1, fe_dim = 512, p_drop=0.1):
        super(WideResNet, self).__init__()        
        #nb_ch = [fe_dim, 256 * widen_factor, 512 * widen_factor, 1024 * widen_factor, 1024 * widen_factor]
        nb_ch = [fe_dim, 256 * widen_factor, 512 * widen_factor, 1024 * widen_factor, 2048 * widen_factor, 2048 * widen_factor]
        self.nb_ch = nb_ch
                
        self.deriv1 = DerivConv1d( input_dim, fe_dim, kernel_size=9, nb_deriv=8)
        self.deriv2 = DerivConv1d( input_dim, fe_dim, kernel_size=9, nb_deriv=8)
        self.deriv3 = DerivConv1d( input_dim, fe_dim, kernel_size=9, nb_deriv=8)
        self.deriv4 = DerivConv1d( input_dim, fe_dim, kernel_size=9, nb_deriv=8)
        self.deriv5 = DerivConv1d( input_dim, fe_dim, kernel_size=9, nb_deriv=8)
        
        self.block1 = wrn_block(nb_ch[0],          nb_ch[1], n, p_drop, stride=1, kernel_size=7, dilation=3)
        self.block2 = wrn_block(nb_ch[1]+nb_ch[0], nb_ch[2], n, p_drop, stride=1, kernel_size=5, dilation=3)
        self.block3 = wrn_block(nb_ch[2]+nb_ch[0], nb_ch[3], n, p_drop, stride=1, kernel_size=3, dilation=2)
        self.block4 = wrn_block(nb_ch[3]+nb_ch[0], nb_ch[4], n, p_drop, stride=1, kernel_size=3, dilation=2)
        self.block5 = wrn_block(nb_ch[4]+nb_ch[0], nb_ch[5], n, p_drop, stride=1, kernel_size=3, dilation=2)
        
        #self.fc2 = nn.Linear(nb_ch[4], output_dim)
        self.fc2 = nn.Linear(nb_ch[5], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        
        x1 = self.deriv1(x)
        x2 = self.deriv2(x)
        x3 = self.deriv3(x)
        x4 = self.deriv4(x)
        x5 = self.deriv5(x)
 
        x = self.block1(x1)
        x = torch.cat([x2, x], 1)
        x = self.block2(x)
        x = torch.cat([x3, x], 1)
        x = self.block3(x)
        x = torch.cat([x4, x], 1)
        x = self.block4(x)
        x = torch.cat([x5, x], 1)
        x = self.block5(x)
  
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        x = sigmoid(x)
        return x

    def predict2(self, x, alpha):
        x = x.permute(0, 2, 1).contiguous()
        
        x1 = self.deriv1(x)
        x2 = self.deriv2(x)
        x3 = self.deriv3(x)
        x4 = self.deriv4(x)
        x5 = self.deriv5(x)
 
        x = self.block1(x1)
        x = torch.cat([x2, x], 1)
        x = self.block2(x)
        x = torch.cat([x3, x], 1)
        x = self.block3(x)
        x = torch.cat([x4, x], 1)
        x = self.block4(x)
        x = torch.cat([x5, x], 1)
        x = self.block5(x)
  
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        x = x * alpha
        x = sigmoid(x)
        return x

class Net_snr(BaseNet):
    def __init__(self, input_dim, output_dim, lr=1e-3, cuda=False, float16=False, single_gpu=True):
        cprint('p', '\n  Net_snr:')
        super(Net_snr, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.create_net()
        self.cuda(cuda, float16, single_gpu)
        self.set_opt(opt='adama', lr=lr, wd_after=5e-5)
        self.lr_sch = NoamLRScheduler(100, 1e-3, 1e-3, 1e-6)
        
    def create_net(self):
        self.model = WideResNet(self.input_dim, self.output_dim)
        self.J = nn.MSELoss()
        print('    nb_params: %.2fM' % ( self.get_nb_parameters() / 1e6 ))

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda, float16=self.float16)        
        n, t, d = x.shape
        
        out = self.model(x)        
        y = y[:,-1,:]
        out = out[:,-1,:]
        loss = self.J(out.view(n,-1),y.view(n,-1))
        self.step(loss)
        return to_float(loss)
        
    def eval(self, x, y):
        x, y = to_variable(var=(x, y), volatile=True, cuda=self.cuda, float16=self.float16)
        n, t, d = x.shape
        
        out = self.model(x)
        out = out[:,-1,:]
        y = y[:,-1,:]
        loss = self.J(out.view(n,-1),y.view(n,-1))
        return to_float(loss)

    def predict(self, x):
        start_move = time.time()
        x,  = to_variable(var=(x, ), volatile=True, cuda=self.cuda, float16=self.float16)   # tensor como entrada directamente
        # print(f'Move time to cuda: {float((time.time()-start_move)*1000)} ms')
        n, t, d = x.shape # sobra
        start = time.time()
        out = self.model(x)  
        end = time.time()
        # print(f'Inference time for {t} windows done at the same time is: {float((end-start)*1000)} ms') 
        # print(f'Time per window is {float((end-start)*1000/t)} ms')         
        return out.data


    def predict2(self, x, alpha):
        x,  = to_variable(var=(x, ), volatile=True, cuda=self.cuda, float16=self.float16)
        n, t, d = x.shape
        out = self.model.predict2(x, alpha)        
        return out.data
    
    def predict_windows(self, x, snrsize, n=4):
        accum_time = 0
        start_move = time.time()
        x,  = to_variable(var=(x, ), volatile=True, cuda=self.cuda, float16=self.float16)
        print(f'Move time to cuda: {float((time.time()-start_move)*1000)} ms')
        b, t, d = x.shape
        snr = torch.zeros(b, t, snrsize).float().to(x)
        x0 = x[:,:1,:].repeat(1, n-1, 1)
        x = torch.cat([x0, x], 1)
        for i in range(n, t):
            xt = x[:,i-n:i,:]
            with torch.no_grad():
                start = time.time()
                out = self.model(xt)
                end = time.time()
                # print(f'Time of {i} set of {n} windows: {end-start}')
                accum_time += end-start
            snr[:,i-n,:] = out[:,-1,:]
        print(f'Total time for {t} windows: {float(accum_time*1000):.5f} ms')
        print(f'Average time per window: {float(accum_time/t*1000):.5f} ms')
        return snr.data

if __name__ == '__main__':
    net = WideResNet(2, 2)
    x = Variable(torch.randn(2, 50,  2))
    x[:,25:,:] = -10000
    y = net(x)
    print(x)
    print(y)

    print(y.size())
