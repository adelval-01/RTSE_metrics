from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division
import numpy as np
import h5py, pickle, sys, os, gzip, time
import soundfile as sf
import scipy.special as sc
from audiolazy.lazy_lpc import lpc 
from scipy import linalg
from scipy import signal

# ----------------------------------------------------------------------------------------------------------------------
def cprint(color, text, supports_color=True, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    if supports_color:
        print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    else:
        print("%s" % text, **kwargs)
    sys.stdout.flush()

# ----------------------------------------------------------------------------------------------------------------------
def read_list_str(file, list_=None):
    if isinstance(file,list):
        list_ = []
        for filei in file:
           list_ = read_list_str(filei, list_=list_)
        return list_
    
    if isinstance(file,str):   
        root,ext = os.path.splitext(file)
        if ext == '.gz':
            with gzip.open(file, 'r') as f:
                return read_list_str(f, list_=list_)
        else:
            with open(file, 'r') as f:
                return read_list_str(f, list_=list_)
    else:
        if list_ is None:
            list_ = []
            
        for line in file.readlines():
            list_.append( line.strip() )
    return list_

def read_list_str_np(file):    
    return np.array(read_list_str(file))

# FE -------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def resample(x, fs0, fs1):
    if fs0 != fs1:
        if fs1 < fs0:
            f = int(fs0 / fs1)
            x = signal.resample_poly(x, 1, f)
        else:
            f = int(fs1 / fs0)
            x = signal.resample_poly(x, f, 1)
    return x

def offset(x):
    xx = np.zeros(len(x))
    for i in range(len(x)): 
        if i == 241:
            xx[i] = x[i] + 0.99899 * xx[i-1]
        elif i > 241:
	        xx[i] = (x[i] - x[i-1]) + 0.99899 * xx[i-1]
        else:
            xx[i] = x[i]     
    return xx
    
def preemphasis(x):
    x0 = x[0] * (1.0 - 0.97)
    for i in reversed(range(len(x))): 
        x[i] = x[i] - x[i-1] * 0.97
    x[0] = x0
    return x   
    
def windowing2(x, fs=16000, Ns=0.040, Ms=0.010):
    N = int(Ns * fs)
    M = int(Ms * fs)
    n = (len(x) + M - 1) // M    
    T = (n - 1) * M + N 
    xa = x.copy()
    if T > len(x):
        xa.resize(T, refcheck=False)
    m = np.arange(0, n * M, M)
    ind = np.arange(N).reshape(-1, 1) + m.reshape(1, -1)
    return xa[ind.astype(int).T].astype(np.float32)

# ----------------------------------------------------------------------------------------------------------------------
def compute_vad(x, fs=16000, N=0.025, M=0.010, nfft=512):  
    frame = int(N * fs)
    shift = int(M * fs)     
    maxamplitude = 0.75
    th = 4
    lambd = 0.6
    n = 4
    x = maxamplitude * (x / (np.abs(x).max()))
    Nframes = int(1 + np.ceil((x.size-shift)/shift))  
    #Nframes = int(np.floor((x.size-(frame-shift))/shift))  #as matlab
    
    x = preemphasis(offset(x))
    X = windowing2(x, fs, N, M) * np.hamming(frame)    
    Xfft = np.abs(np.fft.fft(X, nfft))
    Xfft = Xfft[:, 1:int(nfft/2+1)]
    
    # init
    LTSE = np.zeros(Xfft.shape)
    r_est = np.zeros(Xfft.shape)
    r_lt = np.zeros(Xfft.shape)
    LTSD = np.zeros([Nframes,1])
    vad = np.zeros([Nframes,1])

    LTSE[0] = np.max(Xfft[0:n+1],axis=0)
    suelo = np.ones(int(nfft/2)) * (0.22 * np.power(np.mean(Xfft), 2))
    r_est[0] = np.mean(np.power(Xfft[0:n],2),0)
    r_lt[0] = r_est[0]
    LTSD[0] = 10*np.log10(1/(nfft)*sum(np.power(LTSE[0],2)/r_est[0]))
    vad[0] = int(LTSD[0] > th)

    # LTSE & LTSD => vad
    for j in range(1,Nframes-1):
        a = np.max([j - n, 1])
        aa = np.min([j + n, Nframes + 1])
        LTSE[j] = np.max(Xfft[np.max([j-n, 0]):np.min([j+n, Nframes+1])+1],axis=0)

        if vad[j-1]:
            r_est[j] = r_est[j-1]
        else:
            r_est[j] = lambd*r_est[j-1]+(1-lambd)*np.power(Xfft[j],2)

        r_lt[j] = 0.9*r_lt[j-1]+(1-0.9)*np.min([np.power(Xfft[j],2),r_est[j]],0)
        LTSD[j] = 10*np.log10(1/(nfft)*sum(np.power(LTSE[j],2)/np.max([r_lt[j],suelo],0)))
        vad[j] = int(LTSD[j] > th)

    vad = np.kron(vad, np.ones(shift))
    vad = np.reshape(vad, (1,np.product(vad.shape)))[0]
    vad = vad[0:x.size]
       
    return np.asarray(vad, dtype=np.int16) 

# ----------------------------------------------------------------------------------------------------------------------
def create_snr(x, n, snr=10):
    vad = compute_vad(x)

    pot_x = np.var(x[vad==1])
    x_norm = x/np.sqrt(pot_x)

    pot_n = np.var(n[vad==1])
    n_norm = n/np.sqrt(pot_n * np.power(10,snr/10))

    y = x_norm + n_norm
    y = y/np.abs(np.max(y))
    y[y==0]=1e-7

    snr_real = 10 * np.log10(np.var(x_norm[vad==1])) - 10 * np.log10(np.var(n_norm[vad==1]))
    if (np.abs((snr_real - snr) > 0.1)):
        print('WARNING: real and theoretical SNRs are not the same')

    return y, x_norm, n_norm

# ----------------------------------------------------------------------------------------------------------------------
def create_snr_rand(x, n, snr=10, fs=16000, noiseind1=0, noiseind2=0):
    sx = x.size
    sn = n.size
    if sx < sn:
        if noiseind2 == 0:
            noiseind1 = np.random(sn-sx-1)
            noiseind2 = noiseind1+sx-1
        n = n[noiseind1-1:noiseind2]
    else:
        if sx > sn:
            n = np.broadcast_to(n, (np.ceil(sx/sn), n))
            sn = n.size
            noiseind1 = np.random(sn-sx-1)
            n = n[noiseind1-1:noiseind1+sx-1]
 
    pot_x = np.var(x)
    x_norm = x/np.sqrt(pot_x)

    pot_n = np.var(n)
    n_norm = n/np.sqrt(pot_n * np.power(10,snr/10))

    y = x_norm + n_norm
    y = y/np.abs(np.max(y))

    snr_real = 10 * np.log10(np.var(x_norm)) - 10 * np.log10(np.var(n_norm))
    if (np.abs((snr_real - snr) > 0.1)):
        print('WARNING: real and theoretical SNRs are not the same')

    return y, x_norm, n_norm

# ENHANCEMENT -------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def apply_filter(data, filt, frame=640, shift=160, nfft=1024):

    win = np.sqrt(np.hanning(frame))
    win = np.array(win, dtype=np.float32)
    
    yw = np.zeros(data.size)
    #yw[0:frame] = data[0:frame]*win*1e-3
    it = int(np.floor((data.size-frame)/shift))
    
    start = time.time()
    for i in range(0,it):
        xw = data[i*shift : i*shift+frame] * win
        Xfft = np.fft.fft(xw, nfft)
        Xfft = Xfft[0:int(nfft/2+1)]

        outf = Xfft * filt[:,i]
        fliped = np.flip(np.conj(outf[1:int(nfft/2)]), axis=0)
        outw = np.concatenate((outf, fliped), axis=0)
        outw = np.real(np.fft.ifft(outw, nfft, axis=0))
        yw[i*shift : i*shift+frame] = yw[i*shift : i*shift+frame] + outw[0:frame]*win
    end = time.time()
    print(f'Evaluation time for {it} windows done at the same time is: {float((end-start)*1000):.5f} ms') 
    print(f'Average time per window is {float((end-start)*1000/it):.5f} ms') 
    #yw[it*shift+frame:] = data[it*shift+frame:]*1e-3

    return yw

# ----------------------------------------------------------------------------------------------------------------------
def noiseReduction(data, snr_net, fs, frame, shift, nfft, gmin):
    
    # Add un-audible noise to avoid signals with 0
    data = data + 1e-7*np.random.rand(data.size) 
    
    # Load snr_net
    snr_net = np.concatenate((snr_net, snr_net[int(nfft/2-1):, :]), axis=0)
    
    # VAD
    ini = int(np.floor((nfft*300)/fs))
    out = int(np.ceil((nfft*2500)/fs))
    E = np.mean(snr_net[ini:out,:], axis=0)
    vad = 1/(1 + np.exp(-90*(E-0.05)))

    difference = 1 - snr_net
    difference[difference < 1e-7] = 1e-7
    gamma = 1 / difference
    upsilon = gamma * snr_net
    g = np.exp(0.5 * sc.exp1(upsilon)) * snr_net
    g[g > 1] = 1
    gmin = gmin/2
    gtotal = np.power(g, snr_net) * np.power((gmin * snr_net), (1 - snr_net))
    filt = np.power(gtotal,vad) * np.power(1e-3,1-vad)
    yw = apply_filter(data, filt, frame, shift, nfft)
    yw = np.array(yw, dtype=np.float32)

    return yw, filt

# QUALITY -------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from Alpha04 import tabla_alpha_04_db as dbvals
from Alpha04 import tabla_alpha_04_val as Gvals

def wada_snr(x, fr=16000):
    x = np.asarray(x, dtype=np.float32)
    abs_x = np.abs(x)
    abs_x = [1e-10 if value <= 1e-10 else value for value in abs_x]

    dVal1 = np.mean(abs_x)
    dVal2 = np.mean(np.log(abs_x))
    dEng = np.sum(np.power(abs_x, 2))
    dVal3 = np.log(dVal1) - dVal2

    dSNRix = [index for index, val in enumerate(Gvals) if val < dVal3]
    if len(dSNRix) == 0:
        dSNR = dbvals[0]
    elif len(dSNRix) == len(dbvals):
        dSNR = dbvals[-1]
    else:
        dSNRix = np.max(dSNRix)
        dSNR = dbvals[dSNRix] \
               + (dVal3 - Gvals[dSNRix]) / (Gvals[dSNRix + 1] - Gvals[dSNRix]) \
               * (dbvals[dSNRix + 1] - dbvals[dSNRix])

    dFactor = 10**(dSNR / 10)
    dNoiseEng = dEng / (1 + dFactor)
    dSigEng = dEng * dFactor / (1 + dFactor)

    SNR = 10 * np.log10(dSigEng / dNoiseEng)
    return SNR

# ----------------------------------------------------------------------------------------------------------------------
def snr_sim(x, n, filt, snrini):

    y, x_norm, n_norm = create_snr(x, n, snrini)

    xfilter = apply_filter(x_norm, filt)
    nfilter = apply_filter(n_norm, filt)

    vad = compute_vad(x)

    xpot = np.var(xfilter[vad==1]) 
    npot = np.var(nfilter[vad==1])

    snrpost = 10 * np.log10(xpot/npot)
    snrpre = 10 * np.log10(np.var(x_norm[vad==1])/np.var(n_norm[vad==1]))
    deltasnr = snrpost - snrpre

    return snrpost, snrpre, deltasnr

# ----------------------------------------------------------------------------------------------------------------------
def itakura2(x, y, lpcorder=12, frame=640):
    
    if len(x) > len(y):
        x = x[:len(y)]    
    if len(y) > len(x):
        y = y[:len(x)]
    L = x.size
    shift = int(frame/2)
    
    Nframes = int(1 + np.floor((L-frame)/shift))
    Ex = np.zeros(Nframes)
    Ey = np.zeros(Nframes)
    distIxy = np.zeros(Nframes)
    distIyx = np.zeros(Nframes)
    distISxy = np.zeros(Nframes)
    distISyx = np.zeros(Nframes)

    for i in range(0,Nframes):
        fx = x[i*shift:frame+i*shift]
        fy = y[i*shift:frame+i*shift]
        Ex[i] = np.sum(np.power(fx,2))
        Ey[i] = np.sum(np.power(fy,2))
        Fx = np.correlate(fx, fx, mode='full')/frame
        Rx = linalg.toeplitz(Fx[frame-1:frame+lpcorder])
        Fy = np.correlate(fy, fy, mode='full')/frame
        Ry = linalg.toeplitz(Fy[frame-1:frame+lpcorder])
                
        A = lpc(fx,lpcorder).numlist
        A = np.array(A, dtype=np.float32)
        B = lpc(fy,lpcorder).numlist
        B = np.array(B, dtype=np.float32)

        # Itakura
        pB = np.dot(np.transpose(B),Rx)
        pA = np.dot(np.transpose(A),Rx)
        distIxy[i] = np.log10(np.dot(pB,B)/np.dot(pA,A))
        
        pB = np.dot(np.transpose(B),Ry)
        pA = np.dot(np.transpose(A),Ry)
        distIyx[i] = np.log10(np.dot(pA,A)/np.dot(pB,B))

        distI = (distIxy + distIyx)/2

        # Itakura-Saito
        pB = np.dot(np.transpose(A-B),Rx)
        pA = np.dot(np.transpose(A),Rx)
        distISxy[i] = np.dot(pB,A-B) / np.dot(pA,A)

        pA = np.dot(np.transpose(B-A),Ry)
        pB = np.dot(np.transpose(B),Ry)
        distISyx[i] = np.dot(pA,B-A) / np.dot(pB,B)
        
        distIS = (distISxy + distISyx)/2
    
    distI[Ex<0.05] = 0
    distI = np.mean(distI[distI>0]) 

    distIS[Ex<0.05] = 0
    distIS = np.mean(distIS[distIS>0]) 

    return distI, distIS

